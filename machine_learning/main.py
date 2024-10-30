# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict
from celery.result import AsyncResult
from worker.worker import celery_app, process_txt2img, process_img2img  # Import Celery app and tasks
from loguru import logger
import os
from fastapi.responses import FileResponse
from utils import save_image_to_storage
import shutil

app = FastAPI(title="Image Processing API")

# In-memory storage for job results (for simplicity)
# For production, consider using a persistent database
job_results: Dict[str, Dict] = {}

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

@app.post("/generate-txt2img", response_model=Img2ImgResponse)
def generate_txt2img(request: Txt2ImgRequest):
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
    
    # Enqueue the Celery task to the default queue
    task = process_txt2img.delay(params)
    
    # Store the task ID and initial status
    job_results[job_id] = {'celery_task_id': task.id, 'status': 'queued'}
    
    return Img2ImgResponse(job_id=job_id, status='queued')

class Img2ImgRequest(BaseModel):
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
    strength: float = 0.7  # Strength parameter for img2img (default: 0.7)

@app.post("/generate-img2img", response_model=Img2ImgResponse)
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
    
    # Save the uploaded image to a temporary location
    try:
        temp_dir = 'temp_images'
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
    
    # Enqueue the Celery task
    task = process_img2img.delay(params)
    
    # Store the task ID and initial status
    job_results[job_id] = {'celery_task_id': task.id, 'status': 'queued', 'input_image_path': input_image_path}
    
    return Img2ImgResponse(job_id=job_id, status='queued')

@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    """
    Endpoint to check the status of a submitted job.
    """
    job = job_results.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    # Use celery_app to get AsyncResult
    celery_task = celery_app.AsyncResult(job['celery_task_id'])
    
    if celery_task.state == 'PENDING':
        status = 'queued'
    elif celery_task.state == 'STARTED':
        status = 'processing'
    elif celery_task.state == 'SUCCESS':
        result = celery_task.result
        if result['status'] == 'completed':
            status = 'completed'
            job_results[job_id]['image_path'] = result['image_path']
        else:
            status = 'failed'
            job_results[job_id]['reason'] = result.get('reason', 'Unknown error.')
    elif celery_task.state == 'FAILURE':
        status = 'failed'
        job_results[job_id]['reason'] = str(celery_task.info)
    else:
        status = celery_task.state.lower()
    
    job_results[job_id]['status'] = status
    
    response = {'job_id': job_id, 'status': status}
    if status == 'completed':
        response['image_path'] = job_results[job_id].get('image_path')
    elif status == 'failed':
        response['reason'] = job_results[job_id].get('reason', 'Unknown error.')
    
    return response

@app.get("/download-image/{job_id}")
def download_image(job_id: str):
    """
    Endpoint to download the processed image.
    """
    job = job_results.get(job_id)
    if not job or job['status'] != 'completed':
        raise HTTPException(status_code=404, detail="Image not found or not yet processed.")
    
    image_path = job.get('image_path')
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file does not exist.")
    
    return FileResponse(image_path, media_type='image/png', filename=os.path.basename(image_path))
