from ultralytics import YOLO
from diffusers import (
    StableDiffusionXLInpaintPipeline,
)
from loguru import logger
import numpy as np
import torch
from torch import autocast
from PIL import Image, ImageDraw, ImageFilter

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

# Hand Detailer Class (Using Ultralytics YOLO)
class HandDetailer:
    def __init__(self, device='cuda', shared_components=None, height=1024, width=1024):
        """
        Initialize the HandDetailer using Ultralytics YOLO for hand detection.

        Args:
            device (str): Device to run the model on ('cuda' or 'cpu').
            shared_components (dict): Shared components like tokenizer.
            height (int): Desired height of the output image.
            width (int): Desired width of the output image.
        """
        self.device = device
        self.height = height
        self.width = width
        self.hand_detector = self.load_hand_detector()
        self.shared_components = shared_components  # Will be used when initializing the pipeline
        self.tokenizer = shared_components.get('tokenizer') if shared_components else None
        self.pipe = None  # Pipeline will be initialized in the enhance_hands method

    def load_hand_detector(self):
        """
        Load the pre-trained YOLO hand detection model.

        Returns:
            The loaded YOLO model.
        """
        try:
            # Load the pre-trained YOLOv8 hand detection model
            model = YOLO('/app/models/hand_yolov8s.pt')  # Replace with your actual model path
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error loading YOLO hand detection model: {e}")
            raise

    def initialize_pipeline(self):
        """
        Initialize the inpainting pipeline.
        """
        try:
            if self.shared_components:
                pipe = StableDiffusionXLInpaintPipeline(
                    vae=self.shared_components['vae'],
                    text_encoder=self.shared_components['text_encoder'],
                    text_encoder_2=self.shared_components['text_encoder_2'],
                    tokenizer=self.shared_components['tokenizer'],
                    tokenizer_2=self.shared_components['tokenizer_2'],
                    unet=self.shared_components['unet'],
                    scheduler=self.shared_components['scheduler'],
                    feature_extractor=self.shared_components['feature_extractor'],
                ).to(self.device)
                pipe.enable_xformers_memory_efficient_attention()
                return pipe
            else:
                raise ValueError("Shared components are required for pipeline initialization.")
        except Exception as e:
            logger.error(f"Error initializing inpaint pipeline: {e}")
            raise

    def enhance_hands(self, image, hand_prompt="highly detailed hands, 8k resolution"):
        """
        Enhance detected hands in the image using inpainting.

        Args:
            image (PIL.Image.Image): The input image.
            hand_prompt (str): Prompt for inpainting hands.

        Returns:
            PIL.Image.Image: The image with enhanced hands.
        """
        try:
            # Initialize the pipeline
            self.pipe = self.initialize_pipeline()

            # Truncate the hand prompt if necessary
            if self.tokenizer:
                hand_prompt = truncate_prompt(hand_prompt, self.tokenizer)
            else:
                logger.warning("Tokenizer not found. Skipping prompt truncation for hand_prompt.")

            # Convert PIL Image to RGB and NumPy array
            img_rgb = image.convert("RGB")
            img_np = np.array(img_rgb)

            # Perform hand detection using YOLO
            results = self.hand_detector(img_np)

            # Extract bounding boxes and confidence scores
            boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: (num_boxes, 4)
            scores = results[0].boxes.conf.cpu().numpy()  # shape: (num_boxes,)
            if len(boxes) == 0:
                logger.info("No hands detected.")
                # Clean up
                del self.pipe
                torch.cuda.empty_cache()
                return image

            # Filter boxes based on confidence threshold
            confidence_threshold = 0.5  # Adjust as needed
            filtered_indices = np.where(scores >= confidence_threshold)[0]
            boxes = boxes[filtered_indices]
            if len(boxes) == 0:
                logger.info("No hands detected with sufficient confidence.")
                # Clean up
                del self.pipe
                torch.cuda.empty_cache()
                return image

            # Create mask for hands based on bounding boxes
            mask = self.create_mask_from_boxes(image, boxes)

            # Verify mask size
            if mask.size != image.size:
                logger.warning(f"Mask size {mask.size} does not match image size {image.size}. Resizing mask.")
                mask = mask.resize(image.size, resample=Image.NEAREST)

            # Log image and mask sizes
            logger.debug(f"Image size: {image.size}, Mask size: {mask.size}")

            # Inpaint the masked regions
            with torch.inference_mode():
                with autocast(self.device):
                    result = self.pipe(
                        prompt=hand_prompt,
                        image=image,
                        mask_image=mask,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=self.height,
                        width=self.width
                    )
            enhanced_image = result.images[0]

            # Feather the mask for smoother blending
            blurred_mask = self.feather_mask(mask, radius=5)

            # Blend the inpainted image with the original
            final_image = self.blend_images(image, enhanced_image, blurred_mask)

            # Clean up
            del self.pipe
            torch.cuda.empty_cache()

            return final_image

        except Exception as e:
            logger.exception("Error enhancing hands:")
            raise

    def create_mask_from_boxes(self, image, boxes):
        """
        Create a binary mask from bounding boxes with padding and blurred edges.

        Args:
            image (PIL.Image.Image): The input image in RGB format.
            boxes (np.ndarray): Detected bounding boxes with shape (num_boxes, 4).

        Returns:
            PIL.Image.Image: The binary mask image.
        """
        try:
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)

                # Apply padding
                padding = int(0.05 * max(x2 - x1, y2 - y1))  # Reduced padding
                x1_padded = max(x1 - padding, 0)
                y1_padded = max(y1 - padding, 0)
                x2_padded = min(x2 + padding, image.width)
                y2_padded = min(y2 + padding, image.height)

                # Draw rectangle on mask
                draw.rectangle([x1_padded, y1_padded, x2_padded, y2_padded], fill=255)

                logger.debug(f"Hand detection bounding box: ({x1_padded}, {y1_padded}, {x2_padded}, {y2_padded})")

            # Apply Gaussian blur to mask edges for smoother inpainting
            mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

            return mask

        except Exception as e:
            logger.error(f"Error creating mask from boxes: {e}")
            raise

    def blend_images(self, original, inpainted, mask):
        """
        Blend the original and inpainted images using the mask.

        Args:
            original (PIL.Image.Image): The original image.
            inpainted (PIL.Image.Image): The inpainted image.
            mask (PIL.Image.Image): The blurred mask image.

        Returns:
            PIL.Image.Image: The blended image.
        """
        try:
            blended = Image.composite(inpainted, original, mask)
            return blended
        except Exception as e:
            logger.error(f"Error blending images: {e}")
            raise

    def feather_mask(self, mask, radius=5):
        """
        Apply Gaussian blur to the mask to feather the edges.

        Args:
            mask (PIL.Image.Image): The binary mask image.
            radius (int): Radius for Gaussian blur.

        Returns:
            PIL.Image.Image: The blurred mask.
        """
        try:
            blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=radius))
            return blurred_mask
        except Exception as e:
            logger.error(f"Error feathering mask: {e}")
            raise
