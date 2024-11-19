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


# Enhanced HandDetailer Class with ControlNet and Hand Pose Estimation
class HandDetailer:
    def __init__(
        self,
        device='cuda',
        blur_radius=5,
        height=1024,
        width=1024,
        padding_ratio=0.2,
        timestamp=None
    ):
        """
        Initialize the HandDetailer using MediaPipe Hands, ControlNet, and inpainting.
        """
        self.device = device
        self.blur_radius = blur_radius
        self.height = height
        self.width = width
        self.padding_ratio = padding_ratio
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize YOLO with fixed model path for hand detection
        try:
            self.yolo = YOLO("/app/models/hand_yolov8s.pt").to(self.device)
            logger.info("YOLO model loaded successfully for hand detection.")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

        # Initialize MediaPipe Hands
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7
            )
            logger.info("MediaPipe Hands initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing MediaPipe Hands: {e}")
            raise

        self.tokenizer = None  # Will be set when initializing the pipeline
        self.pipe = None

    def initialize_pipeline(self):
        """
        Initialize the inpainting pipeline using ControlNet compatible with SDXL, using a model for hand detailing.
        """
        try:
            # Load SDXL-compatible ControlNet model for hands
            controlnet = ControlNetModel.from_pretrained(
                "thibaud/controlnet-openpose-sdxl-1.0",  # Ensure this model exists or replace with appropriate
                torch_dtype=torch.float16,
                use_safetensors=False
            ).to(self.device)

            # Load the hand detailing model components from a pre-trained file
            model_file = "/app/models/sdxl/ponyRealism_v22MainVAE.safetensors"  # Update path as needed

            # Load the inpainting pipeline
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
                model_file,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(self.device)

            pipe.enable_xformers_memory_efficient_attention()
            self.tokenizer = pipe.tokenizer

            # Clean up
            torch.cuda.empty_cache()
            gc.collect()

            logger.info("Inpaint pipeline with ControlNet initialized successfully using the hand detailing model.")
            return pipe
        except Exception as e:
            logger.error(f"Error initializing inpaint pipeline: {e}")
            raise

    def match_skin_tones(self, source_image, target_image, mask):
        """
        Adjust the color distribution of the target image's skin region to match that of the source image.
        
        Args:
            source_image (PIL.Image.Image): Original image.
            target_image (PIL.Image.Image): Inpainted image.
            mask (PIL.Image.Image): Mask used for inpainting.
        
        Returns:
            PIL.Image.Image: Color-matched image.
        """
        try:
            # Convert images to NumPy arrays
            source_np = np.array(source_image).astype(np.float32)
            target_np = np.array(target_image).astype(np.float32)
            mask_np = np.array(mask.convert('L')).astype(bool)  # Ensure mask is single channel and boolean

            # Initialize an empty array for the matched skin
            matched_target_np = target_np.copy()

            # Perform histogram matching for each channel separately within the masked region
            for channel in range(3):  # Assuming RGB
                # Extract the source and target skin regions for the current channel
                source_skin = source_np[:, :, channel][mask_np]
                target_skin = target_np[:, :, channel][mask_np]

                if len(source_skin) == 0 or len(target_skin) == 0:
                    logger.warning(f"No skin pixels found in channel {channel} for histogram matching.")
                    continue

                # Compute the histogram matching for the current channel
                # Note: match_histograms expects both input and reference to be images
                # Therefore, we'll create masked images for each channel
                source_channel = source_np[:, :, channel].copy()
                source_channel[~mask_np] = 0  # Mask out non-skin regions
                target_channel = target_np[:, :, channel].copy()
                target_channel[~mask_np] = 0  # Mask out non-skin regions

                # Perform histogram matching on the masked regions
                matched_channel = match_histograms(target_channel, source_channel, channel_axis=None)

                # Replace only the masked regions in the target image
                matched_target_np[:, :, channel][mask_np] = matched_channel[mask_np]

            # Clip values to valid range and convert back to uint8
            matched_target_np = np.clip(matched_target_np, 0, 255).astype(np.uint8)

            # Convert back to PIL Image
            matched_target_image = Image.fromarray(matched_target_np, 'RGB')
            return matched_target_image

        except Exception as e:
            logger.error(f"Error matching skin tones: {e}")
            raise

    def enhance_hands(
        self,
        image,
        hand_prompt="hand",
        hand_negative_prompt="bad_hands, missing_finger"
    ):
        """
        Enhance or fix hands in the provided image using YOLO, MediaPipe Hands, and ControlNet inpainting.
        
        Args:
            image (PIL.Image.Image): Input image.
            hand_prompt (str): Prompt for the inpainting model.
            hand_negative_prompt (str): Negative prompt for the inpainting model.
        
        Returns:
            PIL.Image.Image: Image with enhanced hands.
        """
        try:
            if image is None:
                logger.error("No image provided to enhance_hands.")
                return None

            # Initialize the inpainting pipeline
            self.pipe = self.initialize_pipeline()

            # Check if pipeline is initialized
            if self.pipe is None:
                logger.error("Pipeline is None after initialization.")
                return None

            # Truncate the hand prompt if necessary
            if self.tokenizer:
                hand_prompt = truncate_prompt(hand_prompt, self.tokenizer)
            else:
                logger.warning("Tokenizer not found. Skipping prompt truncation for hand_prompt.")

            # Truncate the negative prompt if necessary
            if hand_negative_prompt and self.tokenizer:
                hand_negative_prompt = truncate_prompt(hand_negative_prompt, self.tokenizer)
            elif hand_negative_prompt:
                logger.warning("Tokenizer not found. Skipping prompt truncation for negative_prompt.")

            # Convert PIL Image to RGB and NumPy array
            img_rgb = image.convert("RGB")
            img_np = np.array(img_rgb)

            # Perform hand detection using YOLOv8
            detections = self.yolo(img_np)  # YOLOv8 inference

            # Filter detections for hands (assuming YOLOv8's 'hand' class is known, adjust as necessary)
            # You might need to adjust the class index based on the YOLO model's classes
            hand_class_index = 0  # Replace with the correct class index if different
            hand_detections = [det for det in detections[0].boxes if det.cls == hand_class_index]

            if not hand_detections:
                logger.info("No hands detected by YOLO.")
                # Clean up resources
                del self.pipe
                torch.cuda.empty_cache()
                gc.collect()
                return image

            logger.info(f"Detected {len(hand_detections)} hand(s) using YOLO.")

            # Create a copy of the original image to paste enhanced hands
            final_image = image.copy()

            # Process each detected hand individually
            for idx, det in enumerate(hand_detections, start=1):
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

                    # Apply padding, ensuring coordinates stay within image boundaries
                    x1_padded = max(x1 - pad_x, 0)
                    y1_padded = max(y1 - pad_y, 0)
                    x2_padded = min(x2 + pad_x, image.width)
                    y2_padded = min(y2 + pad_y, image.height)

                    logger.info(f"Hand {idx} original bbox: ({x1}, {y1}, {x2}, {y2})")
                    logger.info(f"Hand {idx} padded bbox: ({x1_padded}, {y1_padded}, {x2_padded}, {y2_padded})")

                    # Crop the padded hand region from the original image
                    hand_region = img_rgb.crop((x1_padded, y1_padded, x2_padded, y2_padded))

                    # Convert cropped hand to NumPy array for MediaPipe processing
                    hand_np = np.array(hand_region)

                    # Perform hand landmark detection using MediaPipe Hands
                    results = self.hands.process(hand_np)

                    if not results.multi_hand_landmarks:
                        logger.warning(f"No hand landmarks detected for hand {idx}. Skipping.")
                        continue

                    logger.info(f"Detected landmarks for hand {idx}.")

                    # Create a precise mask using the detected hand landmarks
                    hand_mask = self.create_precise_mask_with_landmarks(hand_region, results.multi_hand_landmarks)

                    # Ensure the mask is the same size as the cropped hand region
                    hand_mask = hand_mask.resize(hand_region.size, resample=Image.NEAREST)

                    # Create a control image from the hand landmarks for ControlNet
                    control_image = Image.new('RGB', hand_region.size, (0, 0, 0))
                    for landmarks in results.multi_hand_landmarks:
                        for lm in landmarks.landmark:
                            landmark_x = int(lm.x * hand_region.width)
                            landmark_y = int(lm.y * hand_region.height)
                            # Draw a single white pixel for each landmark
                            control_image.putpixel((landmark_x, landmark_y), (255, 255, 255))

                    # Upscale the hand region, mask, and control image
                    upscale_factor = 5  # Adjust as needed
                    high_res_size = (hand_region.width * upscale_factor, hand_region.height * upscale_factor)
                    hand_region_high_res = hand_region.resize(high_res_size, resample=Image.LANCZOS)
                    hand_mask_high_res = hand_mask.resize(high_res_size, resample=Image.NEAREST)
                    control_image_high_res = control_image.resize(high_res_size, resample=Image.NEAREST)

                    # Inpaint the high-resolution hand
                    with autocast(self.device):
                        result = self.pipe(
                            prompt=hand_prompt,
                            negative_prompt=hand_negative_prompt,
                            image=hand_region_high_res,
                            mask_image=hand_mask_high_res,
                            control_image=control_image_high_res,
                            num_inference_steps=30,
                            guidance_scale=7.5,
                            height=high_res_size[1],
                            width=high_res_size[0]
                        )
                    enhanced_hand_high_res = result.images[0]

                    # Downscale the enhanced hand back to original hand region size
                    enhanced_hand = enhanced_hand_high_res.resize(hand_region.size, resample=Image.LANCZOS)

                    # Paste the enhanced hand back into the original image
                    final_image.paste(enhanced_hand, (x1_padded, y1_padded), hand_mask)

                except Exception as hand_e:
                    logger.error(f"Error processing hand {idx}: {hand_e}")
                    continue  # Continue processing other hands

            # Clean up resources
            try:
                del self.pipe
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Cleaned up inpainting pipeline and cleared GPU cache.")
            except Exception as cleanup_e:
                logger.error(f"Error during cleanup: {cleanup_e}")

            return final_image
        
        except Exception as e:
            logger.error(f"Error processing enhance_hands: {e}")

    def create_precise_mask_with_landmarks(self, image, hand_landmarks):
        """
        Create a precise binary mask using the convex hull of all hand landmarks.
        
        Args:
            image (PIL.Image.Image): The cropped hand image.
            hand_landmarks (list): List of detected hand landmarks from MediaPipe.
        
        Returns:
            PIL.Image.Image: Binary mask image.
        """
        try:
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)

            for landmarks in hand_landmarks:
                # Extract (x, y) coordinates of all landmarks for the current hand
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

            # Calculate the dilation kernel size as 5% of the minimum dimension
            expansion_percentage = 0.05
            min_dim = min(image.width, image.height)
            kernel_size = max(1, int(expansion_percentage * min_dim))

            # Ensure the kernel size is odd and at least 3 for better dilation effect
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 3:
                kernel_size = 3

            # Create a circular (elliptical) structuring element
            kernel = cv2.get_structuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

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
