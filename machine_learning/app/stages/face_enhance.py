from gfpgan import GFPGANer
import numpy as np
from PIL import Image
from loguru import logger


class GFPGANEnhancer:
    def __init__(self, device='cuda'):
        self.device = device
        self.gfpganer = self.load_model()

    def load_model(self):
        try:
            # Initialize GFPGAN with the pretrained model
            model_path = '/app/models/GFPGANv1.3.pth'  # Update with the correct path to the GFPGAN model
            gfpganer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            return gfpganer
        except Exception as e:
            logger.error(f"Error loading GFPGAN model: {e}")
            raise

    def enhance_faces(self, image):
        """
        Enhance faces in the image using GFPGAN.
        """
        try:
            img_np = np.array(image)
            _, _, restored_faces = self.gfpganer.enhance(
                img_np,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            # Convert back to PIL Image
            enhanced_image = Image.fromarray(restored_faces)
            return enhanced_image
        except Exception as e:
            logger.error(f"Error enhancing faces with GFPGAN: {e}")
            raise
