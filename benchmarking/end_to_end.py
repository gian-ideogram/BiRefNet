import sys
sys.path.insert(0, "/home/gianfavero/projects/")
sys.path.insert(0, "/home/gianfavero/projects/BiRefNet/")
sys.path.insert(0, '/home/gianfavero/projects/ideogram-api-python/src')
sys.path.append('/home/gianfavero/projects/tfrecords')

import argparse
import os
import logging
from io import BytesIO
import requests
import io
import base64

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from BiRefNet.benchmarking.factory import get_model
from BiRefNet.ideogram_utils import pil_image_to_bytes, reduce_spill, recover_original_rgba

from PIL import Image
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ideogram_api.client import IdeogramAPI
from ideogram_api.exceptions import IdeogramAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def generate_from_prompt(prompt, custom_model_uri=None, num_images=4) -> dict:
    """
    Generate an image from a prompt using Ideogram API.
    
    Args:
        prompt: The prompt to use to generate the image
        custom_model_uri: The URI of a custom model to use

    Returns:
        dict: Result from Ideogram API generate endpoint
    """
    
    client = IdeogramAPI(timeout=480)
        
    try:
        # Use generate endpoint to generate image
        logger.info("Generating image using Ideogram generate endpoint...")
        logger.info(f"Prompt: {prompt}")
        generate_result = client.generate(
            prompt=prompt,
            custom_model_uri=custom_model_uri,
            magic_prompt="OFF",
            num_images=num_images,
            rendering_speed="TURBO",
            return_image_as_bytes=True,  # Get image bytes directly
        )
        
        if generate_result.get("data"):
            result_url = generate_result['data'][0]['url']
            logger.info(f"Image generation successful! Result URL: {result_url}")
                
    except IdeogramAPIError as e:
        logger.error(f"Ideogram API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

    return [generate_result['data'][i]['image_bytes'] for i in range(len(generate_result['data']))]

def upscale_image(image: Image.Image) -> Image.Image:
    # API endpoint
    url = "https://api.topazlabs.com/image/v1/enhance"

    # Headers
    headers = {
        "X-API-Key": None # TODO: Add API key
    }

    # Save PIL image to buffer as png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Files and form data
    files = {
        "image": buffer
    }
    data = {
        "model": "Text Refine",
        "output_width": 6144,
        "output_height": 6144,
        "output_format": "png"
    }
    response = requests.post(url, headers=headers, files=files, data=data)

    return response

def main(custom_model_uri: str=None, prompt: str=None):
    images_bytes = generate_from_prompt(prompt, custom_model_uri, num_images=1)
    images_pil = [Image.open(io.BytesIO(image_byte)) for image_byte in images_bytes]

    # Upscale images using Topaz Labs API
    response = upscale_image(images_pil[0])
    print(response.json())

if __name__ == "__main__":

    API_KEY = None # TODO: Add API key
    os.environ['IDEOGRAM_API_KEY'] = API_KEY

    custom_model_uri = "model/gian-green-graphic-2k/version/0"
    prompt = "A man on a bike"

    main(custom_model_uri, prompt)