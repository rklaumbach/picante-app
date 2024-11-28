import torch
from torch import autocast
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import gc
import logging
from .unlimited_prompt_optimization import get_weighted_text_embeddings_sdxl



# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def truncate_prompt(prompt, tokenizer, max_length=77):
    """Truncate the prompt to the maximum length supported by the tokenizer."""
    tokens = tokenizer.encode(prompt, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens)

# Image Generator Class
class ImageGenerator:
    def __init__(self, device='cuda', height=1024, width=1024, pipeline=None):
        self.device = device
        self.height = height
        self.width = width
        self.tokenizer = None
        self.pipe = pipeline  # Accept pre-initialized pipeline

        if self.pipe is None:
            logger.info("Pipeline is none in ImageGenerator")
        # if self.pipe is None:
        #     self.initialize_pipeline()

    def initialize_pipeline(self, pipeline_type='txt2img'):
        if self.pipe is not None:
            logger.info(f"Using existing {pipeline_type} pipeline.")
            return self.pipe

        try:
            model_file = "/models/sdxl/ponyRealism_v22MainVAE.safetensors"

            if pipeline_type == 'txt2img':
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_file,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            elif pipeline_type == 'img2img':
                pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_file,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            else:
                raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

            pipe.to(self.device)

            # Set Clip Skip if applicable
            if hasattr(pipe.text_encoder, 'config') and hasattr(pipe.text_encoder.config, 'clip_skip'):
                pipe.text_encoder.config.clip_skip = 2
            if hasattr(pipe.text_encoder_2, 'config') and hasattr(pipe.text_encoder_2.config, 'clip_skip'):
                pipe.text_encoder_2.config.clip_skip = 2

            pipe.enable_xformers_memory_efficient_attention()
            self.tokenizer = pipe.tokenizer

            self.pipe = pipe  # Store the pipeline instance
            logger.info(f"{pipeline_type} pipeline loaded successfully.")
            return self.pipe
        except Exception as e:
            logger.error(f"Error initializing {pipeline_type} pipeline: {e}")
            raise

    def generate_image(
        self,
        image_prompt,
        negative_prompt="...",
        num_inference_steps=30,
        guidance_scale=6.5,
        image=None,          # Optional: PIL.Image.Image, np.ndarray, or torch.Tensor for img2img
        strength=0.3         # Optional: Strength parameter for img2img
    ):
        try:
            # Determine pipeline type based on presence of reference image
            if image is None:
                pipeline_type = 'txt2img'
            else:
                pipeline_type = 'img2img'

            logger.info(f"Using {pipeline_type} pipeline.")
            pipe = self.pipe

            # Generate weighted text embeddings using sd_embed
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = get_weighted_text_embeddings_sdxl(
                pipe,
                prompt=image_prompt,
                neg_prompt=negative_prompt,
            )

            # Prepare pipeline inputs
            pipeline_kwargs = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": self.height,
                "width": self.width
            }

            if pipeline_type == 'img2img':
                if image is None:
                    raise ValueError("Image must be provided for img2img pipeline.")
                pipeline_kwargs["image"] = image
                pipeline_kwargs["strength"] = strength

            logger.info(f"Generating image using {pipeline_type} pipeline.")
            with torch.inference_mode():
                with torch.no_grad(), autocast(self.device):
                    result = pipe(**pipeline_kwargs)

            generated_image = result.images[0]

            # Clean up to free memory
            del prompt_embeds, negative_prompt_embeds
            del pooled_prompt_embeds, negative_pooled_prompt_embeds
            del result
            torch.cuda.empty_cache()
            gc.collect()

            logger.info("Image generation completed.")

            return generated_image
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise

    def unload_pipeline(self, pipeline_type='txt2img'):
        """
        Unloads the specified pipeline to free up GPU memory.
        """
        try:
            if self.pipe is not None:
                logger.info(f"Unloading {pipeline_type} pipeline.")
                del self.pipe
                self.pipe = None
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"{pipeline_type} pipeline unloaded successfully.")
        except Exception as e:
            logger.error(f"Error unloading {pipeline_type} pipeline: {e}")