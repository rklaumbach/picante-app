# app/app.py

import threading
from typing import List, Optional, Dict
import modal
from modal import Function, asgi_app, Queue, Dict as ModalDict, Secret
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request
from pydantic import BaseModel
from uuid import uuid4
from io import BytesIO
from PIL import Image
import os
import shutil
import logging
import json
from supabase import create_client, Client
from datetime import datetime, timezone
from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetInpaintPipeline
from workflows import Txt2ImgFaceDetailUpscaleWorkflow, Img2ImgFaceDetailUpscaleWorkflow
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define CUDA image parameters
cuda_version = "12.4.1"
flavor = "cudnn-devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Initialize the base image using the official NVIDIA CUDA image
image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")

# Install system dependencies required by OpenCV
image = image.run_commands(
    "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git build-essential cmake"
)

# Install Python dependencies from requirements.txt
image = image.pip_install_from_requirements(
    requirements_txt="app/requirements.txt"
)

# **NEW**: Install xformers with CUDA 12.4 compatibility
image = image.run_commands(
    "pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124"
)

image = image.run_commands(
    'python -c "from transformers import utils; utils.move_cache()" || echo "Transformers cache migration skipped."'
    ' && python -c "from ultralytics import YOLO; YOLO.create_settings_file()" || echo "Ultralytics Settings file already exists."'
)

# Consolidate all file copies into a single step by copying the entire 'app/' directory
image = image.copy_local_dir(
    local_path="app",
    remote_path="/app"
)

# Set environment variables
image = image.env({
    "PYTHONUNBUFFERED": "1"
})

# Initialize Modal App with the custom image and attach the secret
supabase = Secret.from_name("supabase")

app = modal.App("image-processing-app", image=image, secrets=[supabase])

# Define Modal Queue for image generation jobs
image_job_queue = Queue.from_name("image-job-queue", create_if_missing=True)

# Define Modal Dict for storing job results
job_results = ModalDict.from_name("job-results", create_if_missing=True)

model_volume = modal.Volume.from_name("model-volume", create_if_missing=True)


# Initialize FastAPI
fastapi_app = FastAPI(title="Image Processing API")

# Define Pydantic Models
class Txt2ImgRequest(BaseModel):
    user_id: str  # Added user_id
    steps: str = 'all'
    upscale_enabled: bool = True
    image_prompt: str
    face_prompt: str = "highly detailed face, 8k resolution"
    negative_prompt: str = (
        "score_4, score_5, score_6, blurry, distortion, lowres, raw, painting(medium), "
        "cartoonized, cartoon, sketch, bad_hands:1.5, bad_proportions, bad_anatomy, "
        "missing_limb, missing_eye, missing_finger:1.5, extra_ears, extra_mouth, "
        "extra_faces, extra_penises, extra_legs, extra_pupils, extra_digits:1.5, "
        "extra_hands:1.5, extra_arms, extra_eyes"
    )
    face_negative_prompt: str = (
        "score_4, score_5, score_6, blurry, distortion, lowres, raw, open_mouth, hands_on_own_face:1.5, hand_on_own_face:1.5 "
        "split_mouth, (child)1.5, facial_mark, cartoonized, cartoon, sketch, "
        "painting(medium), extra_teeth, missing_tooth, missing_teeth, deformed:1.2, "
        "double_chin, mismatched_irises, extra_pupils, no_pupils, mismatched_pupils, "
        "no_sclera, mismatched_sclera, cross_eyed, no_mouth, "
    )
    width: int = 1024
    height: int = 1024
    scaling: int = 2

class ImgResponse(BaseModel):
    job_id: str
    status: str
    width: Optional[int] = None  # Made optional
    height: Optional[int] = None  # Made optional

class JobStatusResponse(BaseModel):
    status: str
    image_urls: Optional[Dict[str, str]] = None
    reason: Optional[str] = None


# Define Pydantic Models for Chat with History
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[ChatMessage]] = []
    max_length: int = 100
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    history: List[ChatMessage]


@app.cls(
    gpu="A100",
    secrets=[supabase],
    container_idle_timeout=60,
    volumes={"/models": model_volume}
)
class Txt2ImgService:
    def __init__(self):
        # Initialize your pipelines here
        self.lock = threading.Lock()
        #self.initialize_pipelines()
        #self.txt2img_pipeline = self.initialize_txt2img_pipeline()
        #self.face_detailer_pipeline = self.initialize_face_detailer_pipeline()
        self.fastapi_app = FastAPI(title="Image Processing API")
        self.fastapi_app.post("/generate-txt2img")(self.generate_txt2img)
        self.fastapi_app.get("/job-status/{job_id}")(self.get_job_status)
        
    @modal.enter()
    def initialize_pipelines(self):
        self.txt2img_pipeline = self.initialize_txt2img_pipeline()
        self.face_detailer_pipeline = self.initialize_face_detailer_pipeline()
        
    def initialize_txt2img_pipeline(self):
        model_file = "/app/models/sdxl/ponyRealism_v22MainVAE.safetensors"

        txt2img_pipeline = StableDiffusionXLPipeline.from_single_file(
            model_file,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )#.to('cuda')

        # Set Clip Skip if applicable
        if hasattr(txt2img_pipeline.text_encoder, 'config') and hasattr(txt2img_pipeline.text_encoder.config, 'clip_skip'):
            txt2img_pipeline.text_encoder.config.clip_skip = 2
        if hasattr(txt2img_pipeline.text_encoder_2, 'config') and hasattr(txt2img_pipeline.text_encoder_2.config, 'clip_skip'):
            txt2img_pipeline.text_encoder_2.config.clip_skip = 2

        return txt2img_pipeline

    def initialize_face_detailer_pipeline(self):
        controlnet_model_path = "/app/models/controlnet/openpose"

        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path,
            torch_dtype=torch.float16,
            use_safetensors=False  # Assuming .safetensors files
        )#.to('cuda')

        model_file = "/app/models/sdxl/ponyRealism_v22MainVAE.safetensors"

        face_detailer_pipeline = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            model_file,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )#.to('cuda')
        return face_detailer_pipeline

    #@modal.web_endpoint(method="GET")
    #@modal.method()
    def get_job_status(self, job_id: str = Query(...)):
        """
        Endpoint to check the status of a submitted job.
        Expects 'job_id' as a query parameter.
        """
        try:
            logger.info(f"GET /job-status called with job_id: {job_id}")

            job = job_results.get(job_id)
            if not job:
                logger.warning(f"Job ID {job_id} not found.")
                raise HTTPException(status_code=404, detail="Job ID not found.")
            
            status = job.get('status', 'unknown')
            response = {'status': status}
            
            if status == 'completed':
                response['image_urls'] = job.get('image_urls', {})
                response['width'] = job.get('width')
                response['height'] = job.get('height')
            elif status == 'failed':
                response['reason'] = job.get('reason', 'Unknown error.')
            
            return JobStatusResponse(**response)
        except Exception as e:
            logger.exception("Error in /job-status:")
            raise HTTPException(status_code=500, detail="Internal Server Error")
    
    async def generate_txt2img(self, request: Txt2ImgRequest):
        """
        Endpoint to submit a Txt2Img image generation job.
        """
        # Create a unique job ID and timestamp
        job_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Prepare parameters
        params = {
            'job_id': job_id,
            'user_id': request.user_id,
            'steps': request.steps,
            'upscale_enabled': request.upscale_enabled,
            'timestamp': timestamp,
            'image_prompt': request.image_prompt,
            'negative_prompt': request.negative_prompt,
            'face_prompt': request.face_prompt,
            'face_negative_prompt': request.face_negative_prompt,
            'height': request.height,
            'width': request.width,
            'device': 'cuda',
            'scaling': request.scaling
        }

        # Since we are in the same container, we can call the method directly
        task = self.process_txt2img.spawn(params)

            # Store the task ID and initial status with timestamp
        job_results[job_id] = {
            'modal_task_id': task.object_id,
            'status': 'queued',
            'image_urls': {},
            'timestamp': timestamp
        }
    
        return ImgResponse(job_id=job_id, status='queued')
    
    @modal.method()
    def process_txt2img(self, job: Dict):
        """
        Modal Function to process Txt2Img image generation jobs.
        """
        with self.lock:

            if self.face_detailer_pipeline is None:
                logger.info("Face Detailer Pipe is none in process_txt2img")

            if self.txt2img_pipeline is None:
                logger.info("Txt2Img Pipe is none in process_txt2img")

            try:
                supabase_admin = get_supabase_client()
                workflow = Txt2ImgFaceDetailUpscaleWorkflow(
                    device=job.get('device', 'cuda'),
                    steps=job.get('steps', 'all'),
                    upscale_enabled=job.get('upscale_enabled', True),
                    timestamp=job.get('timestamp'),
                    image_prompt=job.get('image_prompt', ""),
                    negative_prompt=job.get('negative_prompt'),
                    face_prompt=job.get('face_prompt', ""),
                    face_negative_prompt=job.get('face_negative_prompt', ""),
                    height=job.get('height', 1024),
                    width=job.get('width', 1024),
                    scaling=job.get('scaling'),
                    txt2img_pipeline=self.txt2img_pipeline,
                    face_detailer_pipeline=self.face_detailer_pipeline
                )

                logger.info(f"Scaling in process_txt2img = {job.get('scaling')}")
                
                images_dict = workflow.run()
                
                if images_dict and 'final_image' in images_dict:
                    user_id = job.get('user_id')  # Ensure user_id is part of the job data
                    if not user_id:
                        raise Exception("User ID not provided in job data.")
                    
                    final_image = images_dict['final_image']
                    
                    # Convert PIL Image to bytes
                    img_byte_arr = BytesIO()
                    final_image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    # Define a unique filename, e.g., using timestamp
                    filename = f"{job['timestamp']}_final.png"

                    # Upload to Supabase Storage
                    image_url = upload_image_to_supabase(img_bytes, user_id, filename)

                    upscaled_width = job.get('width')*job.get('scaling') if job.get('upscale_enabled') else job.get('width')
                    upscaled_height = job.get('height')*job.get('scaling') if job.get('upscale_enabled') else job.get('height')

                    # Insert into 'images' table with proper timestamp
                    supabase_admin.from_("images").insert({
                        "user_id": user_id,
                        "image_path": f"{user_id}/{filename}",
                        "filename": filename,
                        "body_prompt": job.get('image_prompt', ''),
                        "face_prompt": job.get('face_prompt', ''),
                        "width": upscaled_width,
                        "height": upscaled_height,
                        "created_at": job.get('timestamp')  # Actual timestamp
                    }).execute()

                    # Update job status
                    job_info = job_results.get(job.get('job_id'))
                    if not job_info:
                        logger.error(f"Job ID {job.get('job_id')} not found in job_results.")
                        raise Exception("Job ID not found in job_results.")
                    
                    job_info['status'] = 'completed'
                    job_info['image_urls'] = {'final_image': image_url}
                    job_results[job.get('job_id')] = job_info

                    logger.info(f"Job {job.get('job_id')} completed successfully.")
                    return {'status': 'completed',
                            'image_urls': {'final_image': image_url},
                            'width': job.get('width')*job.get('scaling'),
                            'height': job.get('height')*job.get('scaling')}
                else:
                    # Update job status to failed
                    job_info = job_results.get(job.get('job_id'))
                    job_info['status'] = 'failed'
                    job_info['reason'] = 'Image generation failed.'
                    job_results[job.get('job_id')] = job_info
                    logger.error(f"Job {job.get('job_id')} failed: Image generation failed.")
                    return {'status': 'failed', 'reason': 'Image generation failed.'}
            except Exception as e:
                # Update job status to failed with reason
                logger.exception("Error processing Txt2Img job:")
                job_id = job.get('job_id', 'unknown')
                job_info = job_results.get(job_id, {})
                job_info['status'] = 'failed'
                job_info['reason'] = str(e)
                job_results[job_id] = job_info
                return {'status': 'failed', 'reason': str(e)}

@app.cls(
    gpu="A100",
    container_idle_timeout=60,
    volumes={"/models": model_volume}
)
class ChatService:
    def __init__(self):
        self.fastapi_app = FastAPI(title="Mistral Chat API")
        self.fastapi_app.post("/chat")(self.generate_chat)
        self.lock = threading.Lock()

    @modal.enter()
    def initialize_model(self):
        self.tokenizer, self.model = self.initialize_chat_model()

    def initialize_chat_model(self):
        MODELS_DIR = "/models/chat"
        MODEL_NAME = "/models/chat/Moistral-11B-v3"  # Replace with your model's path
        MODEL_REVISION = "main"  # Replace with the specific revision if needed

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            cache_dir=MODELS_DIR
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            cache_dir=MODELS_DIR,
            torch_dtype=torch.float16,  # Adjust based on your model
            device_map="auto"
        ).to('cuda')

        return tokenizer, model

    @modal.method()
    def process_chat(self, job: Dict):
        """
        Modal Function to handle chat requests with history.
        """
        try:
            prompt = job.get('prompt', '')
            history = job.get('history', [])
            max_length = job.get('max_length', 100)
            temperature = job.get('temperature', 0.7)

            # Construct the conversation history into the prompt
            conversation = ""
            for message in history:
                role = message.get('role', 'user')
                content = message.get('content', '')
                conversation += f"{role.capitalize()}: {content}\n"
            conversation += f"User: {prompt}\nAssistant:"

            inputs = self.tokenizer.encode(conversation, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id  # To prevent warnings if needed
            )
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the assistant's reply
            assistant_reply = response_text[len(conversation):].strip()

            # Update the history with the new message
            new_history = history.copy()
            new_history.append({"role": "user", "content": prompt})
            new_history.append({"role": "assistant", "content": assistant_reply})

            return {'response': assistant_reply, 'history': new_history}
        except Exception as e:
            logger.exception("Error in chat_endpoint with history:")
            raise HTTPException(status_code=500, detail="Internal Server Error")
    
    # Integrate chat routes into the existing FastAPI app with history
    async def generate_chat(self, request: ChatRequest):
        """
        Endpoint to handle chat requests with history.
        """
        # Prepare parameters
        params = {
            'prompt': request.prompt,
            'history': [message.dict() for message in request.history],
            'max_length': request.max_length,
            'temperature': request.temperature
        }

        # Enqueue the chat job
        task = self.process_chat.spawn(params)

        # Wait for the task to complete and get the result
        result = task.result()

        return ChatResponse(response=result['response'], history=result['history'])


txt2img_service = Txt2ImgService()
chat_service = ChatService()


# Initialize Supabase client using Modal secrets
def get_supabase_client() -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_service_role_key:
        logger.error("Supabase credentials are not set in environment variables.")
        raise Exception("Supabase credentials are missing.")
    supabase_admin: Client = create_client(supabase_url, supabase_service_role_key)
    return supabase_admin

# Define Supabase Upload Function
def upload_image_to_supabase(image_bytes: bytes, user_id: str, filename: str) -> str:
    """
    Uploads image bytes to Supabase Storage and returns the signed URL.
    
    Args:
        image_bytes (bytes): The image data in bytes.
        user_id (str): The ID of the user uploading the image.
        filename (str): The desired filename in Supabase Storage.
    
    Returns:
        str: The signed URL of the uploaded image.
    """
    try:
        supabase_admin = get_supabase_client()
        bucket_name = "user-images"  # Correct bucket name
        object_name = f"{user_id}/{filename}"  # Organize images by user ID

        logger.info(f"Uploading to Supabase Storage: Bucket={bucket_name}, Object={object_name}")

        # Upload the image to Supabase Storage with user_id in metadata
        response = supabase_admin.storage.from_(bucket_name).upload(
            object_name,
            image_bytes,
            {
                "content-type": "image/png",
                "user_id": user_id  # Store user_id in metadata for RLS
            }
        )

        logger.info(f"Upload response: {response}")

        # Verify that 'path' exists in the response to confirm upload success
        if not hasattr(response, 'path') or not response.path:
            logger.error(f"Upload response missing 'path': {response}")
            raise Exception("Upload response missing 'path'.")

        # Generate a signed URL valid for 1 day (86400 seconds)
        signed_url_response = supabase_admin.storage.from_(bucket_name).create_signed_url(
            object_name, 86400
        )

        logger.info(f"Signed URL response: {signed_url_response}")

        # Verify that 'signed_url' exists in the response
        if 'signedURL' not in signed_url_response or not signed_url_response['signedURL']:
            logger.error(f"Failed to generate signed URL: {signed_url_response}")
            raise Exception("Signed URL generation failed.")
        
        # Extract the signed URL
        signed_url = signed_url_response['signedURL']
        logger.info(f"Image uploaded to Supabase Storage and signed URL generated at {signed_url}")
        return signed_url
    except Exception as e:
        logger.exception("Failed to upload image to Supabase Storage:")
        raise

#@fastapi_app.post("/generate-img2img", response_model=ImgResponse)
async def generate_img2img(
    user_id: str = Form(...),  # Required Form field
    steps: str = Form('all'),
    upscale_enabled: bool = Form(True),
    image_prompt: str = Form(...),
    face_prompt: str = Form("highly detailed face, 8k resolution"),
    negative_prompt: str = Form(
        "score_4, score_5, score_6, bad_hands, bad_proportions, bad_anatomy, "
        "missing_limb, missing_eye, missing_finger, extra_ears, extra_mouth, "
        "extra_faces, extra_penises, extra_legs, extra_pupils, extra_digits, "
        "extra_hands, extra_arms, extra_eyes"
    ),
    face_negative_prompt: str = Form(
        "score_4, score_5, score_6, blurry, distortion, lowres, raw, open_mouth, "
        "split_mouth, (child)1.5, facial_mark, cartoonized, cartoon, sketch, "
        "painting(medium), extra_teeth, missing_tooth, missing_teeth, deformed:1.5, "
        "double_chin, mismatched_irises, extra_pupils, no_pupils, mismatched_pupils, "
        "no_sclera, mismatched_sclera, cross_eyed, no_mouth, hands, fingers, hands_on_own_face, hands_on_own_chin"
    ),
    width: int = Form(...),  # Changed from 'res' to 'width'
    height: int = Form(...),  # Added 'height' instead of parsing 'res'
    strength: float = Form(0.7),
    file: UploadFile = File(...)
):
    """
    Endpoint to submit an Img2Img image generation job.
    """
    # Validate width and height
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Width and height must be positive integers.")
    
    # Save the uploaded image to a temporary location within Modal's container
    try:
        temp_dir = '/tmp/temp_images'  # Use /tmp for temporary storage
        os.makedirs(temp_dir, exist_ok=True)
        input_image_path = os.path.join(temp_dir, f"{uuid4()}_{file.filename}")
        
        with open(input_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.exception("Failed to save uploaded image:")
        raise HTTPException(status_code=500, detail="Failed to save uploaded image.")
    finally:
        file.file.close()
    
    # Create a unique job ID and timestamp
    job_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Prepare parameters
    params = {
        'job_id': job_id,
        'user_id': user_id,  # Include user_id
        'steps': steps,
        'upscale_enabled': upscale_enabled,
        'timestamp': timestamp,  # Actual timestamp
        'image_prompt': image_prompt,
        'negative_prompt': negative_prompt,
        'face_prompt': face_prompt,
        'face_negative_prompt': face_negative_prompt,
        'height': height,
        'width': width,
        'strength': strength,
        'device': 'cuda',  # Or make this configurable
        'input_image_path': input_image_path
    }
    
    # Enqueue the Modal Function
    task = process_img2img.spawn(params)
    
    # Store the task ID and initial status with timestamp
    job_results[job_id] = {
        'modal_task_id': task.object_id,
        'status': 'queued',
        'image_urls': {},
        'timestamp': timestamp
    }
    
    return ImgResponse(job_id=job_id, status='queued')

#@fastapi_app.get("/job-status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = job_results.get(job_id)
    if not job:
        logger.warning(f"Job ID {job_id} not found.")
        raise HTTPException(status_code=404, detail="Job ID not found.")

    # Check if the task has completed
    if job['status'] == 'queued' and job['task_future'].done():
        result = job['task_future'].result()
        job['status'] = result.get('status', 'unknown')
        job['image_urls'] = result.get('image_urls', {})
        job['width'] = result.get('width')
        job['height'] = result.get('height')
        if job['status'] == 'failed':
            job['reason'] = result.get('reason', 'Unknown error.')

    response = {'status': job['status']}
    if job['status'] == 'completed':
        response['image_urls'] = job.get('image_urls', {})
        response['width'] = job.get('width')
        response['height'] = job.get('height')
    elif job['status'] == 'failed':
        response['reason'] = job.get('reason', 'Unknown error.')

    return response

# Function to process Img2Img jobs
@app.function(gpu="A100", secrets=[supabase])
def process_img2img(job: Dict):
    """
    Modal Function to process Img2Img image generation jobs.
    """
    try:
        supabase_admin = get_supabase_client()
        workflow = Img2ImgFaceDetailUpscaleWorkflow(
            input_image_path=job.get('input_image_path'),
            device=job.get('device', 'cuda'),
            steps=job.get('steps', 'all'),
            upscale_enabled=job.get('upscale_enabled', True),
            timestamp=job.get('timestamp'),
            image_prompt=job.get('image_prompt', ""),
            negative_prompt=job.get('negative_prompt'),
            face_prompt=job.get('face_prompt', ""),
            face_negative_prompt=job.get('face_negative_prompt', ""),
            height=job.get('height', 1024),
            width=job.get('width', 1024),
            strength=job.get('strength', 0.7),
            scaling=job.get('scaling', 2)
        )
        
        images_dict = workflow.run()
        
        if images_dict and 'final_image' in images_dict:
            user_id = job.get('user_id')  # Ensure user_id is part of the job data
            if not user_id:
                raise Exception("User ID not provided in job data.")
            
            final_image = images_dict['final_image']
            
            # Convert PIL Image to bytes
            img_byte_arr = BytesIO()
            final_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            # Define a unique filename, e.g., using timestamp
            filename = f"{job['timestamp']}_final.png"

            # Upload to Supabase Storage
            image_url = upload_image_to_supabase(img_bytes, user_id, filename)

            upscaled_width = job.get('width')*job.get('scaling') if job.get('upscale_enabled') else job.get('width')
            upscaled_height = job.get('height')*job.get('scaling') if job.get('upscale_enabled') else job.get('height')

            # Insert into 'images' table with proper timestamp
            supabase_admin.from_("images").insert({
                "user_id": user_id,
                "image_path": f"{user_id}/{filename}",
                "filename": filename,
                "body_prompt": job.get('image_prompt', ''),
                "face_prompt": job.get('face_prompt', ''),
                "width": upscaled_width,
                "height": upscaled_height,
                "created_at": job.get('timestamp')  # Actual timestamp
            }).execute()

            # Update job status
            job_info = job_results.get(job.get('job_id'))
            if not job_info:
                logger.error(f"Job ID {job.get('job_id')} not found in job_results.")
                raise Exception("Job ID not found in job_results.")
            
            job_info['status'] = 'completed'
            job_info['image_urls'] = {'final_image': image_url}
            job_results[job.get('job_id')] = job_info

            logger.info(f"Job {job.get('job_id')} completed successfully.")
            return {'status': 'completed',
                    'image_urls': {'final_image': image_url},
                    'width': job.get('width')*job.get('scaling'),
                    'height': job.get('height')*job.get('scaling')}
        else:
            # Update job status to failed
            job_info = job_results.get(job.get('job_id'))
            job_info['status'] = 'failed'
            job_info['reason'] = 'Image generation failed.'
            job_results[job.get('job_id')] = job_info
            logger.error(f"Job {job.get('job_id')} failed: Image generation failed.")
            return {'status': 'failed', 'reason': 'Image generation failed.'}
    except Exception as e:
        # Update job status to failed with reason
        logger.exception("Error processing Img2Img job:")
        job_info = job_results.get(job.get('job_id'))
        if job_info:
            job_info['status'] = 'failed'
            job_info['reason'] = str(e)
            job_results[job.get('job_id')] = job_info
        else:
            logger.error(f"Job ID {job.get('job_id')} not found in job_results.")
        return {'status': 'failed', 'reason': str(e)}

@fastapi_app.get("/verify-installation")
def verify_installation():
    import torch
    import xformers
    return {
        "torch_version": torch.__version__,
        "xformers_version": xformers.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@fastapi_app.get("/verify-cuda-runtime")
def verify_cuda_runtime():
    import ctypes
    try:
        ctypes.CDLL("libnvrtc.so.12")
        ctypes.CDLL("libnvrtc.so")
        return {"status": "CUDA Runtime and NVRTC are correctly installed."}
    except OSError as e:
        return {"status": "Error", "message": str(e)}
    
@fastapi_app.get("/list-nvrtc-libraries")
def list_nvrtc_libraries():
    import os
    nvrtc_paths = []
    for root, dirs, files in os.walk('/'):
        for file in files:
            if file.startswith('libnvrtc.so'):
                nvrtc_paths.append(os.path.join(root, file))
    return {"nvrtc_paths": nvrtc_paths}

# Define Modal Function to serve FastAPI app
@app.function()
@asgi_app()
def serve_txt2img_fastapi():
    return txt2img_service.fastapi_app

@app.function()
@asgi_app()
def serve_chat_fastapi():
    return chat_service.fastapi_app