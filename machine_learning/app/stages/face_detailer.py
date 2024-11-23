# stages/face_detailer.py

from datetime import datetime
import mediapipe as mp
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel
)
import torch
import gc
from loguru import logger
import numpy as np
from skimage.exposure import match_histograms
from PIL import Image, ImageDraw, ImageFilter
import cv2
from torch import autocast
from ultralytics import YOLO
from .unlimited_prompt_optimization import get_weighted_text_embeddings_sdxl

def log_gpu_memory(stage=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    logger.info(f"GPU Memory Usage {stage}:")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved:  {reserved:.2f} GB")


def truncate_prompt(prompt, tokenizer, max_length=77):
    """Truncate the prompt by token count without decoding."""
    try:
        tokens = tokenizer.encode(prompt)
        if len(tokens) > max_length:
            logger.warning("Prompt was truncated to fit the maximum token length.")
            tokens = tokens[:max_length]
            truncated_prompt = tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
            return truncated_prompt
        else:
            return prompt
    except Exception as e:
        logger.error(f"Error truncating prompt: {e}")
        raise

class FaceDetailer:
    def __init__(
        self,
        device='cuda',
        blur_radius=5,
        height=1024,
        width=1024,
        padding_ratio=0.2,
        timestamp=None,
        pipeline=None
    ):
        """
        Initialize the FaceDetailer using MediaPipe Face Mesh, ControlNet, and inpainting.
        """
        self.device = device
        self.blur_radius = blur_radius
        self.height = height
        self.width = width
        self.padding_ratio = padding_ratio
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipe = pipeline  # Accept pre-initialized pipeline

        # Initialize YOLO with fixed model path
        try:
            self.yolo = YOLO("/app/models/face_yolov8m.pt").to(self.device)
            logger.info("YOLO model loaded successfully for face detection.")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        logger.info("Face Mesh Loaded Successfully")

        self.tokenizer = None  # Will be set when initializing the pipeline

        if self.pipe is None:
            self.initialize_pipeline()

    def initialize_pipeline(self):
        """
        Initialize the inpainting pipeline using ControlNet compatible with SDXL.
        """
        try:
            controlnet_model_path="/app/models/controlnet/openpose"

            # Load SDXL-compatible ControlNet model from local files
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path,
                torch_dtype=torch.float16,
                use_safetensors=False  # Assuming you have .safetensors files
            ).to(self.device)

            # Load the face detailing model components
            model_file = "/app/models/sdxl/ponyRealism_v22MainVAE.safetensors"

            # Load the pipeline
            self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
                model_file,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(self.device)

            self.pipe.enable_xformers_memory_efficient_attention()
            self.tokenizer = self.pipe.tokenizer

            # Clean up
            torch.cuda.empty_cache()
            gc.collect()

            logger.info("Inpaint pipeline with ControlNet initialized successfully using the face detailing model.")
        except Exception as e:
            logger.error(f"Error initializing inpaint pipeline: {e}")
            raise

    def enhance_faces(
        self,
        image,
        face_prompt="A highly detailed portrait of a person with clear facial features, 8k resolution",
        face_negative_prompt="score_4, score_5, score_6, raw, open_mouth, split_mouth, (child)1.5, facial_mark, cartoonized, cartoon, lowres, sketch, painting(medium), extra_teeth, missing_tooth, missing_tooth, deformed, double_chin, mismatched_irises, extra_pupils, no_pupils, mismatched_pupils, no_sclera, mismatched_sclera"
    ):
        log_gpu_memory("before face enhancement")
        try:
            if image is None:
                logger.error("No image provided to enhance_faces.")
                return None

            # Generate weighted text embeddings using sd_embed
            (
                face_prompt_embeds,
                face_negative_prompt_embeds,
                face_pooled_prompt_embeds,
                face_negative_pooled_prompt_embeds,
            ) = get_weighted_text_embeddings_sdxl(
                self.pipe,
                prompt=face_prompt,
                neg_prompt=face_negative_prompt,
            )

            # Convert PIL Image to RGB and NumPy array
            img_rgb = image.convert("RGB")
            img_np = np.array(img_rgb)

            # Perform face detection using YOLOv8
            detections = self.yolo(img_np)  # YOLOv8 inference

            # Filter detections for faces (assuming YOLOv8's 'face' class is known, adjust as necessary)
            face_class_index = 0  # Replace with the correct class index if different
            face_detections = [det for det in detections[0].boxes if det.cls == face_class_index]

            if not face_detections:
                logger.info("No faces detected by YOLO.")
                # Clean up resources
                torch.cuda.empty_cache()
                gc.collect()
                return {'enhanced_face': image, 'debug_images': {}}

            logger.info(f"Detected {len(face_detections)} face(s) using YOLO.")

            # Create a copy of the original image to paste enhanced faces
            final_image = image.copy()

            # Initialize a dictionary to hold all debug images
            debug_images = {}

            # Process each detected face individually
            for idx, det in enumerate(face_detections, start=1):
                log_gpu_memory(f"before processing face {idx}")
                try:
                    # Extract bounding box coordinates and ensure they are within image boundaries
                    x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, image.width)
                    y2 = min(y2, image.height)

                    # Calculate padding based on padding_ratio
                    box_width = x2 - x1
                    box_height = y2 - y1
                    pad_x = int(box_width * self.padding_ratio)
                    pad_y = int(box_height * self.padding_ratio)

                    logger.info(f"Face {idx} original bbox: ({x1}, {y1}, {x2}, {y2})")
                    logger.info(f"Face {idx} padded bbox: ({x1 - pad_x}, {y1 - pad_y}, {x2 + pad_x}, {y2 + pad_y})")

                    # Apply padding, ensuring coordinates stay within image boundaries
                    x1_padded = max(x1 - pad_x, 0)
                    y1_padded = max(y1 - pad_y, 0)
                    x2_padded = min(x2 + pad_x, image.width)
                    y2_padded = min(y2 + pad_y, image.height)

                    # Crop the padded face region from the original image
                    face_region = img_rgb.crop((x1_padded, y1_padded, x2_padded, y2_padded))
                    # debug_images[f'face_{idx}_before_landmarks'] = face_region

                    # Convert cropped face to NumPy array for MediaPipe processing
                    face_np = np.array(face_region)

                    # Perform facial landmark detection using MediaPipe Face Mesh
                    results = self.face_mesh.process(face_np)

                    if not results.multi_face_landmarks:
                        logger.warning(f"No facial landmarks detected for face {idx}. Skipping.")
                        continue

                    logger.info(f"Detected landmarks for face {idx}.")

                    # Draw landmarks on the cropped face for debugging
                    img_with_landmarks = face_region.copy()
                    draw = ImageDraw.Draw(img_with_landmarks)
                    for landmarks in results.multi_face_landmarks:
                        for lm in landmarks.landmark:
                            x = int(lm.x * face_region.width)
                            y = int(lm.y * face_region.height)
                            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 0, 0))
                    # debug_images[f'face_{idx}_after_landmarks'] = img_with_landmarks

                    # Create a precise mask using the detected facial landmarks
                    face_mask = self.create_precise_mask_with_landmarks(face_region, results.multi_face_landmarks)
                    face_mask = face_mask.resize(face_region.size, resample=Image.NEAREST)
                    # debug_images[f'face_{idx}_mask'] = face_mask

                    # Create a control image from the facial landmarks for ControlNet
                    control_image = Image.new('RGB', face_region.size, (0, 0, 0))
                    draw_control = ImageDraw.Draw(control_image)
                    for landmarks in results.multi_face_landmarks:
                        for lm in landmarks.landmark:
                            landmark_x = int(lm.x * face_region.width)
                            landmark_y = int(lm.y * face_region.height)
                            # Draw a single white pixel for each landmark
                            control_image.putpixel((landmark_x, landmark_y), (255, 255, 255))
                    # debug_images[f'face_{idx}_control'] = control_image

                    # Upscale the face region, mask, and control image
                    upscale_factor = 5  # You can adjust this factor as needed
                    high_res_size = (face_region.width * upscale_factor, face_region.height * upscale_factor)
                    face_region_high_res = face_region.resize(high_res_size, resample=Image.LANCZOS)
                    face_mask_high_res = face_mask.resize(high_res_size, resample=Image.NEAREST)
                    control_image_high_res = control_image.resize(high_res_size, resample=Image.NEAREST)

                    # debug_images[f'face_{idx}_high_res_before_inpaint'] = face_region_high_res
                    # debug_images[f'face_{idx}_high_res_mask'] = face_mask_high_res
                    # debug_images[f'face_{idx}_high_res_control'] = control_image_high_res

                    # Inpaint the high-resolution face using embeddings
                    with torch.no_grad(), autocast(self.device):
                        result = self.pipe(
                            prompt_embeds=face_prompt_embeds,
                            negative_prompt_embeds=face_negative_prompt_embeds,
                            pooled_prompt_embeds=face_pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=face_negative_pooled_prompt_embeds,
                            image=face_region_high_res,
                            mask_image=face_mask_high_res,
                            control_image=control_image_high_res,
                            num_inference_steps=60,
                            guidance_scale=7.5,
                            strength=0.4,
                            height=high_res_size[1],
                            width=high_res_size[0]
                        )
                    enhanced_face_high_res = result.images[0]
                    # debug_images[f'face_{idx}_high_res_inpainted'] = enhanced_face_high_res

                    # Downscale the enhanced face back to original face region size
                    enhanced_face = enhanced_face_high_res.resize(face_region.size, resample=Image.LANCZOS)
                    # debug_images[f'face_{idx}_enhanced'] = enhanced_face

                    # Paste the enhanced face back into the original image
                    final_image.paste(enhanced_face, (x1_padded, y1_padded), face_mask)

                    log_gpu_memory(f"after processing face {idx}")
                except Exception as e:
                    logger.warning(f"Error processing face {idx}: {e}")
                    continue  # Skip to the next face

                finally:
                    # Delete variables and free memory
                    for var_name in [
                        'face_region', 'face_mask', 'control_image',
                        'face_region_high_res', 'face_mask_high_res', 'control_image_high_res',
                        'enhanced_face_high_res', 'enhanced_face', 'result'
                    ]:
                        if var_name in locals():
                            del locals()[var_name]
                    torch.cuda.empty_cache()
                    gc.collect()

            # Clean up embeddings and memory
            del face_prompt_embeds, face_negative_prompt_embeds, face_pooled_prompt_embeds, face_negative_pooled_prompt_embeds
            torch.cuda.empty_cache()
            gc.collect()

            log_gpu_memory("after face enhancement")

            return {'enhanced_face': final_image, 'debug_images': debug_images}
        except Exception as e:
            logger.exception("Error enhancing faces:")
            raise

    def create_precise_mask_with_landmarks(self, image, face_landmarks):
        """
        Create a precise binary mask using the convex hull of all facial landmarks.

        Args:
            image (PIL.Image.Image): The cropped face image.
            face_landmarks (list): List of detected facial landmarks from MediaPipe.

        Returns:
            PIL.Image.Image: Binary mask image.
        """
        try:
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)

            for landmarks in face_landmarks:
                # Extract (x, y) coordinates of all landmarks for the current face
                points = [
                    (lm.x * image.width, lm.y * image.height)
                    for lm in landmarks.landmark
                ]

                # Convert to NumPy array for OpenCV processing
                points_np = np.array(points, dtype=np.int32)

                # Compute the convex hull
                hull = cv2.convexHull(points_np)

                # Convert back to list of tuples
                hull_points = [tuple(point) for point in hull.squeeze()]

                # Draw the convex hull on the mask
                draw.polygon(hull_points, fill=255)

            # Convert mask to NumPy array for dilation
            mask_np = np.array(mask)

            # Calculate the dilation kernel size as 2% of the minimum dimension
            expansion_percentage = 0.02
            min_dim = min(image.width, image.height)
            kernel_size = max(1, int(expansion_percentage * min_dim))

            # Ensure the kernel size is odd and at least 3 for better dilation effect
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 3:
                kernel_size = 3

            # Create a circular (elliptical) structuring element
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Apply dilation to expand the mask
            mask_np_dilated = cv2.dilate(mask_np, kernel, iterations=1)

            # Convert back to PIL Image
            mask_dilated = Image.fromarray(mask_np_dilated)

            # Apply Gaussian blur for smoother edges
            if self.blur_radius > 0:
                mask_dilated = mask_dilated.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

            return mask_dilated

        except Exception as e:
            logger.error(f"Error creating precise mask with landmarks: {e}")
            raise