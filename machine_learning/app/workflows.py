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
        upscale=True,
        timestamp=None,
        image_prompt="",
        negative_prompt=None,
        face_prompt="",
        face_negative_prompt=None,
        height=1024,
        width=1024
    ):
        """
        Initialize the UltimateWorkflow.
        """
        self.device = device
        self.steps = [step.strip().lower() for step in steps.split(',')]
        self.upscale_enabled = upscale
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_prompt = image_prompt
        self.negative_prompt = negative_prompt
        self.face_prompt = face_prompt
        self.face_negative_prompt = face_negative_prompt
        self.height = height
        self.width = width

        # Setup output directories
        # self.setup_directories()

    # def setup_directories(self):
    #     """
    #     Create necessary directories for saving images.
    #     """
    #     self.output_dirs = {
    #         'txt2img': 'txt2img',
    #         'post_face_fix': 'post_face_fix',
    #         'post_gfpgan': 'post_gfpgan',
    #         'post_hand_fix': 'post_hand_fix',
    #          'upscaled': 'upscaled'
    #      }

    #     for dir_path in self.output_dirs.values():
    #         os.makedirs(dir_path, exist_ok=True)

    def should_run_step(self, selected_steps):
        """
        Determine whether to run a specific step based on the selected workflow steps.
        """
        return any(step in self.steps for step in selected_steps)


    def run(self):
        try:
            image = None

            # Step 1: Image Generation
            if self.should_run_step(['all', 'txt2img']):
                image_generator = ImageGenerator(
                    device=self.device,
                    height=self.height,
                    width=self.width
                )
                image = image_generator.generate_image(
                    self.image_prompt,
                    negative_prompt=self.negative_prompt
                )
                if image is None:
                    logger.error("Image generation failed. Exiting workflow.")
                    return None

                # Save the generated image
                # txt2img_save_path = os.path.join(self.output_dirs['txt2img'], f"{self.timestamp}_txt2img.png")
                # image.save(txt2img_save_path)
                # logger.info(f"Image saved at {txt2img_save_path}")

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
                if image is None:
                    logger.error("No image available for face enhancement. Exiting workflow.")
                    return None

                face_detailer = FaceDetailer(
                    device=self.device,
                    height=self.height,
                    width=self.width,
                    timestamp=self.timestamp
                )
                image = face_detailer.enhance_faces(
                    image,
                    face_prompt=self.face_prompt,
                    face_negative_prompt=self.face_negative_prompt
                )
                if image is None:
                    logger.error("Face enhancement failed. Exiting workflow.")
                    return None

                # Save the face-enhanced image
                # face_fix_save_path = os.path.join(self.output_dirs['post_face_fix'], f"{self.timestamp}_post_face_fix.png")
                # image.save(face_fix_save_path)
                # logger.info(f"Image saved at {face_fix_save_path}")

                # Unload the face_detailer pipeline
                if hasattr(face_detailer, 'pipe') and face_detailer.pipe is not None:
                    del face_detailer.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the face_detailer instance
                del face_detailer
                torch.cuda.empty_cache()
                gc.collect()

            # Step 4: Upscaling
            if self.upscale_enabled and self.should_run_step(['all', 'upscale']):
                upscaler = RealESRGAN(scale = 2)
                if image is None:
                    logger.error("No image available for upscaling. Exiting workflow.")
                    return None

                logger.info("Starting upscaling process with RealESRGAN.")
                image = upscaler.upscale(image)
                if image is None:
                    logger.error("Upscaling failed. Exiting workflow.")
                    return None

                # Save the upscaled image
                # upscaled_save_path = os.path.join(self.output_dirs['upscaled'], f"{self.timestamp}_upscaled.png")
                # image.save(upscaled_save_path)
                # logger.info(f"Upscaled image saved at {upscaled_save_path}")

                # Unload the upscaler pipeline
                if hasattr(upscaler, 'pipe') and upscaler.pipe is not None:
                    del upscaler.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the upscaler instance
                del upscaler
                torch.cuda.empty_cache()
                gc.collect()

            # Return the final image
            return image

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
        upscale=True,
        timestamp=None,
        image_prompt="",
        negative_prompt=None,
        face_prompt="",
        face_negative_prompt=None,
        height=1024,
        width=1024,
        strength=0.7  # Default strength for img2img
    ):
        self.input_image_path = input_image_path
        self.device = device
        self.steps = [step.strip().lower() for step in steps.split(',')]
        self.upscale_enabled = upscale
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_prompt = image_prompt
        self.negative_prompt = negative_prompt
        self.face_prompt = face_prompt
        self.face_negative_prompt = face_negative_prompt
        self.height = height
        self.width = width
        self.strength = strength

        # Setup output directories
        # self.setup_directories()

    # def setup_directories(self):
    #     """
    #     Create necessary directories for saving images.
    #     """
    #     self.output_dirs = {
    #         'img2img': 'img2img',
    #         'post_face_fix': 'post_face_fix',
    #         'post_gfpgan': 'post_gfpgan',
    #         'post_hand_fix': 'post_hand_fix',
    #         'upscaled': 'upscaled'
    #     }

    #     for dir_path in self.output_dirs.values():
    #         os.makedirs(dir_path, exist_ok=True)

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
            # Load the input image
            if self.should_run_step(['all', 'img2img']):
                input_image = self.load_input_image()
            else:
                input_image = None

            image = None

            # Step 1: Image Generation (Img2Img)
            if self.should_run_step(['all', 'img2img']):
                logger.info("Starting Img2Img image generation.")
                image_generator = ImageGenerator(
                    device=self.device,
                    height=self.height,
                    width=self.width
                )
                image = image_generator.generate_image(
                    image_prompt=self.image_prompt,
                    negative_prompt=self.negative_prompt,
                    image=input_image,
                    strength=self.strength,
                    num_inference_steps=24,  # Adjust as needed
                    guidance_scale=6.5        # Adjust as needed
                )
                if image is None:
                    logger.error("Image generation (img2img) failed. Exiting workflow.")
                    return None

                # Save the generated image
                # img2img_save_path = os.path.join(self.output_dirs['img2img'], f"{self.timestamp}_img2img.png")
                # image.save(img2img_save_path)
                # logger.info(f"Image saved at {img2img_save_path}")

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
                if image is None:
                    logger.error("No image available for face enhancement. Exiting workflow.")
                    return None

                face_detailer = FaceDetailer(
                    device=self.device,
                    height=self.height,
                    width=self.width,
                    timestamp=self.timestamp
                )
                image = face_detailer.enhance_faces(
                    image,
                    face_prompt=self.face_prompt,
                    face_negative_prompt=self.face_negative_prompt
                )
                if image is None:
                    logger.error("Face enhancement failed. Exiting workflow.")
                    return None

                # # Save the face-enhanced image
                # face_fix_save_path = os.path.join(self.output_dirs['post_face_fix'], f"{self.timestamp}_post_face_fix.png")
                # image.save(face_fix_save_path)
                # logger.info(f"Image saved at {face_fix_save_path}")

                # Unload the face_detailer pipeline
                if hasattr(face_detailer, 'pipe') and face_detailer.pipe is not None:
                    del face_detailer.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the face_detailer instance
                del face_detailer
                torch.cuda.empty_cache()
                gc.collect()

            # Step 4: Upscaling
            if self.upscale_enabled and self.should_run_step(['all', 'upscale']):
                upscaler = RealESRGAN(scale = 2)
                if image is None:
                    logger.error("No image available for upscaling. Exiting workflow.")
                    return None

                logger.info("Starting upscaling process with RealESRGAN.")
                image = upscaler.upscale(image)
                if image is None:
                    logger.error("Upscaling failed. Exiting workflow.")
                    return None

                # # Save the upscaled image
                # upscaled_save_path = os.path.join(self.output_dirs['upscaled'], f"{self.timestamp}_upscaled.png")
                # image.save(upscaled_save_path)
                # logger.info(f"Upscaled image saved at {upscaled_save_path}")

                # Unload the upscaler pipeline
                if hasattr(upscaler, 'pipe') and upscaler.pipe is not None:
                    del upscaler.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the upscaler instance
                del upscaler
                torch.cuda.empty_cache()
                gc.collect()

            # Return the final image
            return image

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise