import glob
import sys
sys.path.insert(0, "/home/gianfavero/projects/")
sys.path.insert(0, "/home/gianfavero/projects/BiRefNet/")
sys.path.insert(0, '/home/gianfavero/projects/ideogram-api-python/src')
sys.path.append('/home/gianfavero/projects/tfrecords')

import os
import logging
import time
import requests
import io
import torch
import random
from typing import List, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from BiRefNet.benchmarking.factory import get_model
from BiRefNet.ideogram_utils import revert_alpha_blend, fast_foreground_estimation

from PIL import Image
import numpy as np

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

def bg_removal_transform(sample): # from BiRefNet
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1024, 1024)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = sample["images"][0] # PIL.Image
    input_image = transform_pipeline(image) # Tensor (C, H, W) in the range [0.0, 1.0]
    return image, input_image

def collate_fn(batch):
    images = [item[0] for item in batch]
    input_images = [item[1] for item in batch]
    input_images = torch.stack(input_images)
    return {"images": images, "input_images": input_images}

def tile_images(images: List[Image.Image], tile_size=1024, overlap=32) -> Tuple[List, List]:
    """Tile and batch process a list of images."""
    tiles = []
    tile_positions = []
    stride = tile_size - overlap
    
    for image_idx, image in enumerate(images):
        # Convert image to RGB and numpy array
        image = image.convert("RGB")
        img_np = np.array(image)
        h, w = img_np.shape[0], img_np.shape[1]
        
        # Create tiles with overlap for this image
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Extract tile, handling boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile_np = img_np[y:y_end, x:x_end]
                
                # Convert numpy array to PIL Image
                tile_pil = Image.fromarray(tile_np)
                
                # Apply transform (expects dict with "images" key)
                tile = bg_removal_transform({"images": [tile_pil]})
                tiles.append(tile)
                tile_positions.append((image_idx, x, y, x_end - x, y_end - y))  # (image_idx, x, y, width, height)
    
    return tiles, tile_positions

def bg_removal_tiles(tiles: List[Image.Image], model):
    """Perform background removal on a large image by processing it in tiles."""

    # Make a dataloader for the tiles
    dataloader = DataLoader(tiles, batch_size=4, shuffle=False, collate_fn=collate_fn)

    start_time = time.time()
    output_list = evaluate(model, dataloader)
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")

    return output_list

def create_feather_mask(width, height, feather_size):
    """Create a feather mask with soft edges for blending"""
    mask = np.ones((height, width), dtype=np.float32)
    
    # Apply feathering on all edges
    for i in range(feather_size):
        alpha = i / feather_size
        # Top edge
        if i < height:
            mask[i, :] = np.minimum(mask[i, :], alpha)
        # Bottom edge
        if height - 1 - i >= 0:
            mask[height - 1 - i, :] = np.minimum(mask[height - 1 - i, :], alpha)
        # Left edge
        if i < width:
            mask[:, i] = np.minimum(mask[:, i], alpha)
        # Right edge
        if width - 1 - i >= 0:
            mask[:, width - 1 - i] = np.minimum(mask[:, width - 1 - i], alpha)
    
    return mask

def stitch_tiles(output_list: List[Image.Image], tile_positions: List[Tuple[int, int, int, int, int]], 
                 original_images: List[Image.Image], overlap: int) -> List[Image.Image]:
    """
    Stitch the output tiles back together with blending.
    
    Args:
        output_list: List of processed tile images (RGBA)
        tile_positions: List of tuples (image_idx, x, y, width, height) for each tile
        original_images: List of original images to determine output dimensions
        overlap: Overlap in pixels between tiles
    
    Returns:
        List of stitched images (one per original image)
    """
    # Group tiles by image_idx
    num_images = len(original_images)
    stitched_images = []
    
    feather_size = max(overlap // 2, 1)  # Feather half of the overlap region
    
    for image_idx in range(num_images):
        # Get tiles and positions for this image
        image_tiles = []
        image_positions = []
        for i, (idx, x, y, tile_w, tile_h) in enumerate(tile_positions):
            if idx == image_idx:
                image_tiles.append(output_list[i])
                image_positions.append((x, y, tile_w, tile_h))
        
        if not image_tiles:
            continue
        
        # Get original image dimensions
        original_img = original_images[image_idx]
        h, w = original_img.size[1], original_img.size[0]  # PIL uses (width, height)
        
        # Initialize output arrays
        output_image = np.zeros((h, w, 4), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Stitch tiles for this image
        for (x, y, tile_w, tile_h), output in zip(image_positions, image_tiles):
            # Convert PIL image to numpy array
            tile_array = np.array(output).astype(np.float32)
            
            # Create feather mask for this tile
            feather_mask = create_feather_mask(tile_w, tile_h, feather_size)
            
            # Apply the feather mask to the tile
            for c in range(4):  # RGBA channels
                output_image[y:y+tile_h, x:x+tile_w, c] += tile_array[:, :, c] * feather_mask
            
            # Accumulate weights
            weight_map[y:y+tile_h, x:x+tile_w] += feather_mask
        
        # Normalize by the weight map to get the final blended result
        for c in range(4):
            output_image[:, :, c] = np.divide(
                output_image[:, :, c], 
                weight_map, 
                out=np.zeros_like(output_image[:, :, c]), 
                where=weight_map != 0
            )
        
        # Convert back to PIL Image
        stitched_image = Image.fromarray(output_image.astype(np.uint8), mode='RGBA')
        stitched_images.append(stitched_image)
    
    logger.info(f"Stitched {len(output_list)} tiles into {len(stitched_images)} images with {overlap}px overlap")
    
    return stitched_images

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
            is_transparent=False,
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
        "X-API-Key": "3e6edd1f-5f64-4aff-90d9-b82abb4d7418" # TODO: Add API key
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
        "output_width": 4096,
        "output_height": 4096,
        "output_format": "png"
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    
    # Check response status
    if response.status_code != 200:
        logger.error(f"API request failed with status {response.status_code}")
        logger.error(f"Response text: {response.text[:500] if response.text else 'Empty response'}")
        response.raise_for_status()
    
    # The API returns the upscaled image as binary PNG data, not JSON
    # Convert response content to PIL Image
    upscaled_image = Image.open(io.BytesIO(response.content))
    logger.info(f"Image upscaled successfully. Size: {upscaled_image.size}")
    
    return upscaled_image


@torch.no_grad()
def evaluate(model, dataloader):
    torch.set_float32_matmul_precision(['high', 'highest'][0])

    images_list = []
    masks_list = []
    for batch in dataloader:
        input_images = batch["input_images"].to(model.device).half() # needs to be full precision for rmbgv2
        images = batch["images"]

        masks = model(input_images)
        masks[masks < 0.25] = 0

        images_list.extend(images)
        masks_list.append(masks.detach().cpu())
    masks_list = torch.cat(masks_list, dim=0)

    output_list = []
    for image, mask in zip(images_list, masks_list):
        mask = transforms.ToPILImage()(mask)
        mask = mask.resize(image.size)

        image = revert_alpha_blend(image, mask)
        image = fast_foreground_estimation(image, mask, r=90)

        image.putalpha(mask)

        output_list.append(image)

    return output_list

def main(custom_model_uri: str=None, prompt: str=None, model=None):
    """
    Main pipeline: Generate image -> Upscale -> Tiled background removal
    
    Args:
        custom_model_uri: URI of custom Ideogram model to use
        prompt: Text prompt for image generation
        model: Background removal model
    """
    total_start_time = time.time()
    
    # Step 1: Generate image
    # logger.info("=" * 60)
    # logger.info("Step 1: Generating image from prompt...")
    step1_start = time.time()
    # images_bytes = generate_from_prompt(prompt, custom_model_uri, num_images=4)
    # images_pil = [Image.open(io.BytesIO(image_bytes)) for image_bytes in images_bytes]
    step1_time = time.time() - step1_start
    # logger.info(f"Generated image with size: {images_pil[0].size}")
    # logger.info(f"Step 1 time: {step1_time:.2f} seconds ({step1_time/60:.2f} minutes)")
    # os.makedirs("generated_images", exist_ok=True)
    # for i, img in enumerate(images_pil):
    #     rand_int = random.randint(0, 1000000)
    #     img.save(f"generated_images/generated_image_{rand_int}.png")

    # Step 2: Upscale image
    # logger.info("=" * 60)
    # logger.info("Step 2: Upscaling image...")
    step2_start = time.time()
    # upscaled_images = [upscale_image(img) for img in images_pil]
    step2_time = time.time() - step2_start
    # logger.info(f"Upscaled image size: {upscaled_images[0].size}")
    # logger.info(f"Step 2 time: {step2_time:.2f} seconds ({step2_time/60:.2f} minutes)")
    # os.makedirs("upscaled_images", exist_ok=True)
    # for i, img in enumerate(upscaled_images):
    #     rand_int = random.randint(0, 1000000)
    #     img.save(f"upscaled_images/upscaled_image_{rand_int}.png")

    # Step 3: Tiled background removal
    upscaled_images = [Image.open(f) for f in glob.glob("to_upscale/*.png")]
    logger.info("=" * 60)
    logger.info("Step 3: Performing tiled background removal...")
    step3_start = time.time()
    tiled_input, tile_positions = tile_images(upscaled_images, tile_size=2048, overlap=32)
    tiled_output = bg_removal_tiles(tiled_input, model)
    stitched_images = stitch_tiles(tiled_output, tile_positions, upscaled_images, overlap=32)
    step3_time = time.time() - step3_start
    logger.info(f"Step 3 time: {step3_time:.2f} seconds ({step3_time/60:.2f} minutes)")
    
    # Save the final result(s)
    save_start = time.time()
    os.makedirs("final_output", exist_ok=True)
    for i, img in enumerate(stitched_images):
        rand_int = random.randint(0, 1000000)
        img.save(f"final_output/final_output_{rand_int}.png")
    save_time = time.time() - save_start
    logger.info(f"Final output saved to: final_output (save time: {save_time:.2f}s)")
    
    # Total time summary
    total_time = time.time() - total_start_time
    logger.info("=" * 60)
    logger.info("TIMING SUMMARY:")
    logger.info(f"  Step 1 (Generate):     {step1_time:>8.2f}s ({step1_time/total_time*100:>5.1f}%)")
    logger.info(f"  Step 2 (Upscale):      {step2_time:>8.2f}s ({step2_time/total_time*100:>5.1f}%)")
    logger.info(f"  Step 3 (BG Removal):   {step3_time:>8.2f}s ({step3_time/total_time*100:>5.1f}%)")
    logger.info(f"  Save:                  {save_time:>8.2f}s ({save_time/total_time*100:>5.1f}%)")
    logger.info(f"  {'TOTAL':<20} {total_time:>8.2f}s ({total_time/60:>6.2f} minutes)")
    logger.info("=" * 60)
    
if __name__ == "__main__":

    API_KEY = "doGxpC8NIq50eB_6AVZ6dhaOoBFQ-1GxTX2F0KR1UHqWty0ale_QF-xNvU2IH7GR7JpE4HMWKPT0DlpVBKXl4g" # TODO: Add API key
    os.environ['IDEOGRAM_API_KEY'] = API_KEY

    custom_model_uri = "model/gian-green-graphic-2k/version/0"
    prompt = "A graphic design of Mr. Incredible holding a car over his head with the text 'Saving America'"

    model_name = "custom"
    path_to_weight = "green_1e-5_cosine_matting_step_43186.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, device=device, path_to_weight=path_to_weight)

    main(custom_model_uri, prompt, model)