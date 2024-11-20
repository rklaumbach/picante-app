# workflows.py

import torch
from loguru import logger
from datetime import datetime
import os
import warnings
import gc
from stages.image_generator import ImageGenerator
from stages.hand_detailer import HandDetailer
from stages.face_detailer import FaceDetailer
from stages.face_enhance import GFPGANEnhancer
from stages.upscaler import RealESRGAN
from PIL import Image

# Suppress the specific deprecated warning from protobuf
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")


class Txt2ImgFaceDetailUpscaleWorkflow:
    def __init__(
        self,
        device='cuda',
        steps='all',
        upscale_enabled=True,
        timestamp=None,
        image_prompt="",
        negative_prompt=None,
        face_prompt="",
        face_negative_prompt=None,
        height=1024,
        width=1024,
        scaling=2
    ):
        """
        Initialize the Txt2ImgFaceDetailUpscaleWorkflow.
        """
        self.device = device
        self.steps = [step.strip().lower() for step in steps.split(',')]
        self.upscale_enabled = upscale_enabled
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_prompt = image_prompt
        self.negative_prompt = negative_prompt
        self.face_prompt = face_prompt
        self.face_negative_prompt = face_negative_prompt
        self.height = height
        self.width = width
        self.scaling = scaling

    def should_run_step(self, selected_steps):
        """
        Determine whether to run a specific step based on the selected workflow steps.
        """
        return any(step in self.steps for step in selected_steps)


    def run(self):
        try:
            images = {}

            # Step 1: Image Generation
            if self.should_run_step(['all', 'txt2img']):
                image_generator = ImageGenerator(
                    device=self.device,
                    height=self.height,
                    width=self.width
                )
                generated_image = image_generator.generate_image(
                    self.image_prompt,
                    negative_prompt=self.negative_prompt
                )
                if generated_image is None:
                    logger.error("Image generation failed. Exiting workflow.")
                    return None

                images['txt2img_result'] = generated_image

                # Unload the txt2img pipeline to free up GPU memory
                if hasattr(image_generator, 'pipe') and image_generator.pipe is not None:
                    del image_generator.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the image_generator instance
                del image_generator
                torch.cuda.empty_cache()
                gc.collect()

            # Step 2: Face Enhancement
            if self.should_run_step(['all', 'face_fix']):
                if 'txt2img_result' in images:
                    face_input_image = images['txt2img_result']
                else:
                    logger.error("No image available for face enhancement. Exiting workflow.")
                    return None

                face_detailer = FaceDetailer(
                    device=self.device,
                    height=self.height,
                    width=self.width,
                    timestamp=self.timestamp
                )
                face_results = face_detailer.enhance_faces(
                    face_input_image,
                    face_prompt=self.face_prompt,
                    face_negative_prompt=self.face_negative_prompt
                )
                if face_results is None:
                    logger.error("Face enhancement failed. Exiting workflow.")
                    return None

                images['face_detailer_result'] = face_results.get('enhanced_face', face_input_image)
                images['debug_images'] = face_results.get('debug_images', {})

                # Unload the face_detailer pipeline
                if hasattr(face_detailer, 'pipe') and face_detailer.pipe is not None:
                    del face_detailer.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the face_detailer instance
                del face_detailer
                torch.cuda.empty_cache()
                gc.collect()

            # Step 3: Upscaling
            if self.upscale_enabled and self.should_run_step(['all', 'upscale']):
                if 'face_detailer_result' in images:
                    upscaling_input_image = images['face_detailer_result']
                elif 'txt2img_result' in images:
                    upscaling_input_image = images['txt2img_result']
                else:
                    logger.error("No image available for upscaling. Exiting workflow.")
                    return None

                upscaler = RealESRGAN(scale=self.scaling)
                upscaled_image = upscaler.upscale(upscaling_input_image)
                if upscaled_image is None:
                    logger.error("Upscaling failed. Exiting workflow.")
                    return None

                images['upscaled'] = upscaled_image

                # Unload the upscaler pipeline
                if hasattr(upscaler, 'pipe') and upscaler.pipe is not None:
                    del upscaler.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the upscaler instance
                del upscaler
                torch.cuda.empty_cache()
                gc.collect()

            # Return the dictionary of all images
            return images

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise
        finally:
            # Final cleanup to ensure all references are deleted
            torch.cuda.empty_cache()
            gc.collect()


class Img2ImgFaceDetailUpscaleWorkflow:
    def __init__(
        self,
        input_image_path,
        device='cuda',
        steps='all',
        upscale_enabled=True,
        timestamp=None,
        image_prompt="",
        negative_prompt=None,
        face_prompt="",
        face_negative_prompt=None,
        height=1024,
        width=1024,
        strength=0.7,
        scaling=2
    ):
        """
        Initialize the Img2ImgFaceDetailUpscaleWorkflow.
        """
        self.input_image_path = input_image_path
        self.device = device
        self.steps = [step.strip().lower() for step in steps.split(',')]
        self.upscale_enabled = upscale_enabled
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_prompt = image_prompt
        self.negative_prompt = negative_prompt
        self.face_prompt = face_prompt
        self.face_negative_prompt = face_negative_prompt
        self.height = height
        self.width = width
        self.strength = strength
        self.scaling = scaling

    def should_run_step(self, selected_steps):
        """
        Determine whether to run a specific step based on the selected workflow steps.
        """
        return any(step in self.steps for step in selected_steps)

    def load_input_image(self):
        """
        Load the input image from the provided path.
        """
        try:
            image = Image.open(self.input_image_path).convert("RGB")
            logger.info(f"Loaded input image from {self.input_image_path}")
            return image
        except Exception as e:
            logger.error(f"Failed to load input image: {e}")
            raise

    def run(self):
        try:
            images = {}

            # Step 1: Image Generation (Img2Img)
            if self.should_run_step(['all', 'img2img']):
                input_image = self.load_input_image()
                images['input_image'] = input_image

                image_generator = ImageGenerator(
                    device=self.device,
                    height=self.height,
                    width=self.width
                )
                generated_image = image_generator.generate_image(
                    image_prompt=self.image_prompt,
                    negative_prompt=self.negative_prompt,
                    image=input_image,
                    strength=self.strength,
                    num_inference_steps=24,  # Adjust as needed
                    guidance_scale=6.5        # Adjust as needed
                )
                if generated_image is None:
                    logger.error("Image generation (img2img) failed. Exiting workflow.")
                    return None

                images['img2img_result'] = generated_image

                # Unload the img2img pipeline to free up GPU memory
                if hasattr(image_generator, 'pipe') and image_generator.pipe is not None:
                    del image_generator.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the image_generator instance
                del image_generator
                torch.cuda.empty_cache()
                gc.collect()

            # Step 2: Face Enhancement
            if self.should_run_step(['all', 'face_fix']):
                if 'img2img_result' in images:
                    face_input_image = images['img2img_result']
                elif 'input_image' in images:
                    face_input_image = images['input_image']
                else:
                    logger.error("No image available for face enhancement. Exiting workflow.")
                    return None

                face_detailer = FaceDetailer(
                    device=self.device,
                    height=self.height,
                    width=self.width,
                    timestamp=self.timestamp
                )
                face_results = face_detailer.enhance_faces(
                    face_input_image,
                    face_prompt=self.face_prompt,
                    face_negative_prompt=self.face_negative_prompt
                )
                if face_results is None:
                    logger.error("Face enhancement failed. Exiting workflow.")
                    return None

                images['face_detailer_result'] = face_results.get('enhanced_face', face_input_image)
                images['debug_images'] = face_results.get('debug_images', {})

                # Unload the face_detailer pipeline
                if hasattr(face_detailer, 'pipe') and face_detailer.pipe is not None:
                    del face_detailer.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the face_detailer instance
                del face_detailer
                torch.cuda.empty_cache()
                gc.collect()

            # Step 3: Upscaling
            if self.upscale_enabled and self.should_run_step(['all', 'upscale']):
                if 'face_detailer_result' in images:
                    upscaling_input_image = images['face_detailer_result']
                elif 'img2img_result' in images:
                    upscaling_input_image = images['img2img_result']
                elif 'input_image' in images:
                    upscaling_input_image = images['input_image']
                else:
                    logger.error("No image available for upscaling. Exiting workflow.")
                    return None

                upscaler = RealESRGAN(scale=self.scaling)
                upscaled_image = upscaler.upscale(upscaling_input_image)
                if upscaled_image is None:
                    logger.error("Upscaling failed. Exiting workflow.")
                    return None

                images['upscaled'] = upscaled_image

                # Unload the upscaler pipeline
                if hasattr(upscaler, 'pipe') and upscaler.pipe is not None:
                    del upscaler.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the upscaler instance
                del upscaler
                torch.cuda.empty_cache()
                gc.collect()

            # Return the dictionary of all images
            return images

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise
        finally:
            # Final cleanup to ensure all references are deleted
            torch.cuda.empty_cache()
            gc.collect()
