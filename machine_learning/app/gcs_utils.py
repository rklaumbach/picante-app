# app/gcs_utils.py

import os
import json
import logging
from io import BytesIO
from google.cloud import storage
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_gcs_client():
    """
    Initialize and return a Google Cloud Storage client using service account credentials.
    """
    try:
        service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        project = service_account_info.get('project_id')
        client = storage.Client(credentials=credentials, project=project)
        return client
    except Exception as e:
        logger.exception("Failed to initialize GCS client:")
        raise

def upload_image_to_gcs(image: Image.Image, filename: str, bucket_name: str = "picante-ml-image-store") -> str:
    """
    Uploads a PIL Image to GCS and returns the public URL.

    Args:
        image (PIL.Image.Image): The image to upload.
        filename (str): The desired filename in GCS.
        bucket_name (str): The name of the GCS bucket.

    Returns:
        str: The public URL of the uploaded image.
    """
    try:
        # Serialize the image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_bytes = img_byte_arr.read()

        # Initialize the GCS client
        client = initialize_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)

        # Upload the image bytes
        blob.upload_from_string(img_bytes, content_type='image/png')

        # Make the blob publicly accessible (optional)
        blob.make_public()

        logger.info(f"Image uploaded to GCS at {blob.public_url}")
        return blob.public_url
    except Exception as e:
        logger.exception("Failed to upload image to GCS:")
        raise
