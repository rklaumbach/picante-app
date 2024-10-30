# app/app.py

from typing import List, Optional
import modal
from modal import Function, asgi_app, Queue, Dict, Secret
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Security, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from uuid import uuid4
from io import BytesIO
from PIL import Image
import os
import shutil
import logging
import json
from google.cloud import storage
from google.oauth2 import service_account  # Added import


from workflows import Txt2ImgFaceDetailUpscaleWorkflow, Img2ImgFaceDetailUpscaleWorkflow

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

# Consolidate all file copies into a single step by copying the entire 'app/' directory
image = image.copy_local_dir(
    local_path="app",
    remote_path="/app"
)

# Install Python dependencies from requirements.txt
image = image.pip_install_from_requirements(
    requirements_txt="app/requirements.txt"
)

# **NEW**: Install xformers with CUDA 12.4 compatibility
image = image.run_commands(
    "pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124"
)

# Set environment variables
image = image.env({
    "PYTHONUNBUFFERED": "1"
})

# Initialize Modal App with the custom image
app = modal.App("image-processing-app", image=image)

# Define Modal Queue for image generation jobs
image_job_queue = Queue.from_name("image-job-queue", create_if_missing=True)

# Define Modal Dict for storing job results
job_results = Dict.from_name("job-results", create_if_missing=True)

# Initialize FastAPI
fastapi_app = FastAPI(title="Image Processing API")

# Define Pydantic Models
class Txt2ImgRequest(BaseModel):
    steps: str = 'all'
    upscale: bool = True
    image_prompt: str
    face_prompt: str = "highly detailed face, 8k resolution"
    negative_prompt: str = (
        "score_4, score_5, score_6, bad_hands, bad_proportions, bad_anatomy, "
        "missing_limb, missing_eye, missing_finger, extra_ears, extra_mouth, "
        "extra_faces, extra_penises, extra_legs, extra_pupils, extra_digits, "
        "extra_hands, extra_arms, extra_eyes"
    )
    face_negative_prompt: str = (
        "score_4, score_5, score_6, blurry, distortion, lowres, raw, open_mouth, "
        "split_mouth, (child)1.5, facial_mark, cartoonized, cartoon, sketch, "
        "painting(medium), extra_teeth, missing_tooth, missing_teeth, deformed, "
        "double_chin, mismatched_irises, extra_pupils, no_pupils, mismatched_pupils, "
        "no_sclera, mismatched_sclera, cross_eyed, no_mouth, "
    )
    res: str = '1024x1024'

class Img2ImgResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    status: str
    image_urls: Optional[List[str]] = None
    reason: Optional[str] = None

# Initialize GCS Secret
gcs_secret = Secret.from_name("image-store-acc")

def upload_image_to_gcs(image_bytes: bytes, filename: str) -> str:
    """
    Uploads image bytes to GCS and returns the public URL.

    Args:
        image_bytes (bytes): The image data in bytes.
        filename (str): The desired filename in GCS.

    Returns:
        str: The public URL of the uploaded image.
    """
    try:
        # Access the service account JSON from environment variable
        service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        #project = service_account_info.get('project_id')  # Ensure project_id is present

        # Initialize the client
        client = storage.Client(credentials=credentials, project='picante-ml')
        bucket_name = "picante-ml-image-store"  # Your bucket name
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        
        # Upload the image bytes
        blob.upload_from_string(image_bytes, content_type='image/png')
        
        # Generate a signed URL valid for 1 hour (3600 seconds)
        signed_url = blob.generate_signed_url(
            version='v4',
            expiration=3600,  # 1 hour
            method='GET'
        )
        
        logger.info(f"Image uploaded to GCS and signed URL generated at {signed_url}")
        return signed_url
    except Exception as e:
        logger.exception("Failed to upload image to GCS:")
        raise

# Define FastAPI routes
@fastapi_app.post("/generate-txt2img", response_model=Img2ImgResponse)
async def generate_txt2img(request: Txt2ImgRequest):
    """
    Endpoint to submit a Txt2Img image generation job.
    """
    # Parse resolution
    try:
        width, height = map(int, request.res.lower().split('x'))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid resolution format. Use WIDTHxHEIGHT, e.g., 1024x1024.")
    
    # Create a unique job ID
    job_id = str(uuid4())
    
    # Prepare parameters
    params = {
        'steps': request.steps,
        'upscale': request.upscale,
        'timestamp': job_id,  # Using job_id as timestamp for uniqueness
        'image_prompt': request.image_prompt,
        'negative_prompt': request.negative_prompt,
        'face_prompt': request.face_prompt,
        'face_negative_prompt': request.face_negative_prompt,
        'height': height,
        'width': width,
        'device': 'cuda'  # Or make this configurable
    }
    
    # Enqueue the Modal Function
    task = process_txt2img.spawn(params)
    
    # Store the task ID and initial status
    job_results[job_id] = {'modal_task_id': task.object_id, 'status': 'queued', 'image_urls': []}
    
    return Img2ImgResponse(job_id=job_id, status='queued')

@fastapi_app.post("/generate-img2img", response_model=Img2ImgResponse)
async def generate_img2img(
    steps: str = Form('all'),
    upscale: bool = Form(True),
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
        "painting(medium), extra_teeth, missing_tooth, missing_teeth, deformed, "
        "double_chin, mismatched_irises, extra_pupils, no_pupils, mismatched_pupils, "
        "no_sclera, mismatched_sclera, cross_eyed, no_mouth, "
    ),
    res: str = Form('1024x1024'),
    strength: float = Form(0.7),
    file: UploadFile = File(...)
):
    """
    Endpoint to submit an Img2Img image generation job.
    """
    # Validate resolution
    try:
        width, height = map(int, res.lower().split('x'))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid resolution format. Use WIDTHxHEIGHT, e.g., 1024x1024.")
    
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
    
    # Create a unique job ID
    job_id = str(uuid4())
    
    # Prepare parameters
    params = {
        'steps': steps,
        'upscale': upscale,
        'timestamp': job_id,  # Using job_id as timestamp for uniqueness
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
    
    # Store the task ID and initial status
    job_results[job_id] = {'modal_task_id': task.object_id, 'status': 'queued', 'image_urls': []}
    
    return Img2ImgResponse(job_id=job_id, status='queued')

@fastapi_app.get("/job-status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Endpoint to check the status of a submitted job.
    """
    job = job_results.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    # Retrieve the Modal Function Call
    try:
        task = Function.get_call(job['modal_task_id'])
    except Exception as e:
        logger.error(f"Error retrieving task: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    
    if task.done:
        result = task.get()
        if result['status'] == 'completed':
            # Avoid duplicating URLs if already added
            if result['image_url'] not in job_results[job_id]['image_urls']:
                job_results[job_id]['image_urls'].append(result['image_url'])
            job_results[job_id]['status'] = 'completed'
        elif result['status'] == 'failed':
            job_results[job_id]['status'] = 'failed'
            job_results[job_id]['reason'] = result.get('reason', 'Unknown error.')
    
    status = job_results[job_id]['status']
    
    response = {'status': status}
    if status == 'completed':
        response['image_urls'] = job_results[job_id].get('image_urls', [])
    elif status == 'failed':
        response['reason'] = job_results[job_id].get('reason', 'Unknown error.')
    
    return response

# Removed the /download-image/{job_id} endpoint as images are now in GCS

# Define Modal Functions for processing jobs

# Function to process Txt2Img jobs
@app.function(gpu="A100", secrets=[gcs_secret])
def process_txt2img(job):
    """
    Modal Function to process Txt2Img image generation jobs.
    """
    try:
        workflow = Txt2ImgFaceDetailUpscaleWorkflow(
            device=job.get('device', 'cuda'),
            steps=job.get('steps', 'all'),
            upscale=job.get('upscale', True),
            timestamp=job.get('timestamp'),
            image_prompt=job.get('image_prompt', ""),
            negative_prompt=job.get('negative_prompt'),
            face_prompt=job.get('face_prompt', ""),
            face_negative_prompt=job.get('face_negative_prompt', ""),
            height=job.get('height', 1024),
            width=job.get('width', 1024)
        )
        
        final_image = workflow.run()
        
        if final_image:
            # Verify the image before serialization
            logger.info(f"Generated image mode: {final_image.mode}, size: {final_image.size}")
            if final_image.mode != 'RGB':
                final_image = final_image.convert('RGB')
                logger.info("Converted image to RGB mode.")
            
            # Serialize the image to bytes
            img_byte_arr = BytesIO()
            final_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img_bytes = img_byte_arr.read()
            
            # Optionally, save locally for debugging
            final_image.save("/tmp/debug_final_image.png")
            
            # Define the filename for GCS
            filename = f"generated_images/{job['timestamp']}.png"
            
            # Upload the image to GCS and get the URL
            gcs_url = upload_image_to_gcs(img_bytes, filename)
            
            # Store the GCS URL in the Modal Dict
            image_dict = Dict.from_name("image_dict")
            image_dict[job['timestamp']] = gcs_url
            
            # Update job status and store the image URL
            job_results_dict = Dict.from_name("job-results")
            job_info = job_results_dict[job['timestamp']]
            job_info['status'] = 'completed'
            job_info['image_urls'].append(gcs_url)
            job_results_dict[job['timestamp']] = job_info
            
            logger.info(f"Job {job['timestamp']} completed successfully.")
            return {'status': 'completed', 'image_url': gcs_url}
        else:
            # Update job status to failed
            job_results_dict = Dict.from_name("job-results")
            job_info = job_results_dict[job['timestamp']]
            job_info['status'] = 'failed'
            job_info['reason'] = 'Image generation failed.'
            job_results_dict[job['timestamp']] = job_info
            logger.error(f"Job {job['timestamp']} failed: Image generation failed.")
            return {'status': 'failed', 'reason': 'Image generation failed.'}
    except Exception as e:
        # Update job status to failed with reason
        logger.exception("Error processing Txt2Img job:")
        job_results_dict = Dict.from_name("job-results")
        job_info = job_results_dict[job['timestamp']]
        job_info['status'] = 'failed'
        job_info['reason'] = str(e)
        job_results_dict[job['timestamp']] = job_info
        return {'status': 'failed', 'reason': str(e)}

# Function to process Img2Img jobs
@app.function(gpu="A100", secrets=[gcs_secret])
def process_img2img(job):
    """
    Modal Function to process Img2Img image generation jobs.
    """
    try:
        workflow = Img2ImgFaceDetailUpscaleWorkflow(
            input_image_path=job.get('input_image_path'),
            device=job.get('device', 'cuda'),
            steps=job.get('steps', 'all'),
            upscale=job.get('upscale', True),
            timestamp=job.get('timestamp'),
            image_prompt=job.get('image_prompt', ""),
            negative_prompt=job.get('negative_prompt'),
            face_prompt=job.get('face_prompt', ""),
            face_negative_prompt=job.get('face_negative_prompt', ""),
            height=job.get('height', 1024),
            width=job.get('width', 1024),
            strength=job.get('strength', 0.7)
        )
        
        final_image = workflow.run()
        
        if final_image:
            # Serialize the image to bytes
            img_byte_arr = BytesIO()
            final_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Define the filename for GCS
            filename = f"generated_images/{job['timestamp']}.png"
            
            # Upload the image to GCS and get the URL
            gcs_url = upload_image_to_gcs(img_bytes, filename)
            
            # Store the GCS URL in the Modal Dict
            image_dict = Dict.from_name("image_dict")
            image_dict[job['timestamp']] = gcs_url
            
            # Update job status and store the image URL
            job_results_dict = Dict.from_name("job-results")
            job_info = job_results_dict[job['timestamp']]
            job_info['status'] = 'completed'
            job_info['image_urls'].append(gcs_url)
            job_results_dict[job['timestamp']] = job_info
            
            logger.info(f"Job {job['timestamp']} completed successfully.")
            return {'status': 'completed', 'image_url': gcs_url}
        else:
            # Update job status to failed
            job_results_dict = Dict.from_name("job-results")
            job_info = job_results_dict[job['timestamp']]
            job_info['status'] = 'failed'
            job_info['reason'] = 'Image generation failed.'
            job_results_dict[job['timestamp']] = job_info
            logger.error(f"Job {job['timestamp']} failed: Image generation failed.")
            return {'status': 'failed', 'reason': 'Image generation failed.'}
    except Exception as e:
        # Update job status to failed with reason
        logger.exception("Error processing Img2Img job:")
        job_results_dict = Dict.from_name("job-results")
        job_info = job_results_dict[job['timestamp']]
        job_info['status'] = 'failed'
        job_info['reason'] = str(e)
        job_results_dict[job['timestamp']] = job_info
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
def serve_fastapi():
    return fastapi_app
