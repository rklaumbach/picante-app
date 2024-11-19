import torch
from loguru import logger
from datetime import datetime
import argparse
import os
import warnings
import gc
from app.stages.image_generator import ImageGenerator
from app.stages.hand_detailer import HandDetailer
from app.stages.face_detailer import FaceDetailer
from app.stages.face_enhance import GFPGANEnhancer
from app.stages.upscaler import RealESRGANUpscaler


# Suppress the specific deprecated warning from protobuf
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

# Configure logging
logger.add("workflow.log", rotation="1 MB", level="DEBUG")  # Set to DEBUG for detailed logs


def truncate_prompt(prompt, tokenizer, max_length=77):
    """
    Truncate the prompt by token count without decoding.
    """
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


# Ultimate Workflow Class
class UltimateWorkflow:
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
        self.setup_directories()


    def setup_directories(self):
        """
        Create necessary directories for saving images.
        """
        self.output_dirs = {
            'txt2img': 'txt2img',
            'post_face_fix': 'post_face_fix',
            'post_gfpgan': 'post_gfpgan',  # Add this line
            'post_hand_fix': 'post_hand_fix',
            'upscaled': 'upscaled'
        }

        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def should_run_step(self, selected_steps):
        """
        Determine whether to run a specific step based on the selected workflow steps.
        """
        return any(step in self.steps for step in selected_steps)

    def save_image(self, image, stage):
        """
        Save the image to the appropriate directory with a timestamped filename.
        """
        dir_name = self.output_dirs.get(stage)
        if not dir_name:
            logger.warning(f"No directory configured for stage '{stage}'. Skipping save.")
            return
        filename = f"{self.timestamp}_{stage}.png"
        save_path = os.path.join(dir_name, filename)
        try:
            image.save(save_path)
            logger.info(f"Saved {stage} image to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save {stage} image: {e}")

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
                txt2img_save_path = os.path.join(self.output_dirs['txt2img'], f"{self.timestamp}_txt2img.png")
                image.save(txt2img_save_path)
                logger.info(f"Image saved at {txt2img_save_path}")

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
                    negative_prompt=self.face_negative_prompt
                )
                if image is None:
                    logger.error("Face enhancement failed. Exiting workflow.")
                    return None

                # Save the face-enhanced image
                face_fix_save_path = os.path.join(self.output_dirs['post_face_fix'], f"{self.timestamp}_post_face_fix.png")
                image.save(face_fix_save_path)
                logger.info(f"Image saved at {face_fix_save_path}")

                # Unload the face_detailer pipeline
                if hasattr(face_detailer, 'pipe') and face_detailer.pipe is not None:
                    del face_detailer.pipe
                    torch.cuda.empty_cache()
                    gc.collect()

                # Delete the face_detailer instance
                del face_detailer
                torch.cuda.empty_cache()
                gc.collect()

            
            # Step 3: Face Enhancement with GFPGAN
            # if self.should_run_step(['all', 'gfpgan']):
            #     if image is None:
            #         logger.error("No image available for GFPGAN enhancement. Exiting workflow.")
            #         return None

            #     gfpgan_enhancer = GFPGANEnhancer(device=self.device)
            #     image = gfpgan_enhancer.enhance_faces(image)

            #     if image is None:
            #         logger.error("GFPGAN enhancement failed. Exiting workflow.")
            #         return None

            #     # Save the GFPGAN-enhanced image
            #     gfpgan_save_path = os.path.join(self.output_dirs['post_gfpgan'], f"{self.timestamp}_post_gfpgan.png")
            #     image.save(gfpgan_save_path)
            #     logger.info(f"Image saved at {gfpgan_save_path}")

            #     # Delete the GFPGAN enhancer instance
            #     del gfpgan_enhancer
            #     torch.cuda.empty_cache()
            #     gc.collect()

            # Step 4: Upscaling
            if self.upscale_enabled and self.should_run_step(['all', 'upscale']):
                self.upscaler = RealESRGANUpscaler(device=self.device)
                if image is None:
                    logger.error("No image available for upscaling. Exiting workflow.")
                    return None

                logger.info("Starting upscaling process with RealESRGAN.")
                image = self.upscaler.upscale(image)
                if image is None:
                    logger.error("Upscaling failed. Exiting workflow.")
                    return None

                # Save the upscaled image
                upscaled_save_path = os.path.join(self.output_dirs['upscaled'], f"{self.timestamp}_upscaled.png")
                image.save(upscaled_save_path)
                logger.info(f"Upscaled image saved at {upscaled_save_path}")


            # Return the final image
            return image

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise


# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Ultimate Image Generation Workflow")
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help="Specify workflow steps to run separated by commas. Options: all, txt2img, face_fix, hand_fix, upscale. Example: face_fix,upscale"
    )
    parser.add_argument(
        '--disable-upscale',
        action='store_true',
        help="Disable the upscaling step."
    )
    parser.add_argument(
        '--image_prompt',
        type=str,
        required=True,
        help="Prompt for image generation."
    )
    parser.add_argument(
        '--face_prompt',
        type=str,
        default="highly detailed face, 8k resolution",
        help="Specific prompt for face enhancement."
    )
    parser.add_argument(
        '--res',
        type=str,
        default='1024x1024',
        choices=['1280x768', '768x1280', '1024x1024'],
        help="Set the resolution for image generation. Options: 1280x768, 768x1280, 1024x1024. Default: 1024x1024"
    )

    parser.add_argument(
        '--negative_prompt',
        type=str,
        default="score_4, score_5, score_6, bad_hands, bad_proportions, bad_anatomy, missing_limb, missing_eye, missing_finger, extra_ears, extra_mouth, extra_faces, extra_penises, extra_legs, extra_pupils, extra_digits, extra_hands, extra_arms, extra_eyes",
        help="Negative prompt for image generation to avoid certain features."
    )
    parser.add_argument(
        '--face_negative_prompt',
        type=str,
        default="score_4, score_5, score_6, blurry, distortion, lowres, raw, open_mouth, split_mouth, (child)1.5, facial_mark, cartoonized, cartoon, sketch, painting(medium), extra_teeth, missing_tooth, missing_teeth, deformed, double_chin, mismatched_irises, extra_pupils, no_pupils, mismatched_pupils, no_sclera, mismatched_sclera, cross_eyed, no_mouth, ",
        help="Negative prompt for face enhancement to avoid certain features."
    )

    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.res.lower().split('x'))
    except ValueError:
        logger.error(f"Invalid resolution format: {args.res}. Expected format WIDTHxHEIGHT, e.g., 1024x1024.")
        raise

    # Collect a single timestamp for all saved files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize the workflow with command-line arguments
    workflow = UltimateWorkflow(
        device='cuda',
        steps=args.steps,
        upscale=not args.disable_upscale,
        timestamp=timestamp,
        image_prompt=args.image_prompt,
        negative_prompt=args.negative_prompt,
        face_prompt=args.face_prompt,
        face_negative_prompt=args.face_negative_prompt,
        height=height,
        width=width
    )


    try:
        final_image = workflow.run()
        if final_image:
            # Decide where to save the final image based on whether upscaling was performed
            steps_list = [step.strip().lower() for step in args.steps.split(',')]
            if workflow.upscale_enabled and ('upscale' in steps_list or 'all' in steps_list):
                final_save_dir = workflow.output_dirs['upscaled']
                final_stage = 'upscaled'
            elif 'hand_fix' in steps_list or 'all' in steps_list:
                final_save_dir = workflow.output_dirs['post_hand_fix']
                final_stage = 'post_hand_fix'
            elif 'face_fix' in steps_list or 'all' in steps_list:
                final_save_dir = workflow.output_dirs['post_face_fix']
                final_stage = 'post_face_fix'
            else:
                final_save_dir = workflow.output_dirs['txt2img']
                final_stage = 'txt2img'

            filename = f"{timestamp}_{final_stage}_final.png"
            save_path = os.path.join(final_save_dir, filename)
            final_image.save(save_path)
            logger.info(f"Final image saved as {save_path}")
        else:
            logger.info("No final image to save.")
    except Exception as e:
        logger.error(f"An error occurred during the workflow: {e}")