import io
import os
import cairosvg
import PIL
from PIL import Image
import random
import colorsys
import numpy as np
import cv2
from PIL import ImageFilter

def pil_image_to_bytes(pil_image: PIL.Image.Image) -> bytes:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="png")
    return buffer.getvalue()

def rasterize_svg_to_pil(
    svg_bytes: bytes,
    width: int = 1024,
    height: int = 1024,
) -> Image.Image:

    # Rasterize the SVG to a PNG
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=width,
        output_height=height,
        background_color='none'
    )

    # Convert the PNG bytes to a PIL image
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    return img

def check_for_useful_alpha(img: Image.Image, threshold: float = 0.9) -> bool: # do this before cropping to visible area
    alpha = img.getchannel("A")
    iw, ih = img.size
    total_pixels = iw * ih

    alpha_data = np.array(alpha, dtype=np.float32) / 255.0  # normalize to 0–1
    visible_fraction = np.sum(alpha_data > 0) / total_pixels  # average opacity across all pixels
    
    return visible_fraction < threshold and visible_fraction > 0

def crop_to_visible_area(img: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()

    if bbox is None:
        # Fully transparent image → nothing to crop
        return img

    # Crop to the visible bounding box
    cropped_img = img.crop(bbox)

    return cropped_img

def resize_image_for_square(
    img: Image.Image,
    h_w: int = 256,
    fraction: float = 0.7,
) -> Image.Image:
    '''
    Resize an image to be placed in a square of size h_w, centered on the image. 
    The aspect ratio of the image is preserved. Image will be resized only if
    the image is larger than the square.
    '''
    lw, lh = img.size
    if lh > lw:
        target_h = int(fraction * h_w)
        scale = target_h / lh
        if scale < 1:
            target_w = int(lw * scale)
            img_resized = img.resize((target_w, target_h), Image.LANCZOS)
        else:
            img_resized = img
    elif lw >= lh:
        target_w = int(fraction * h_w)
        scale = target_w / lw
        if scale < 1:
            target_h = int(lh * scale)
            img_resized = img.resize((target_w, target_h), Image.LANCZOS)
        else:
            img_resized = img

    return img_resized

def paste_fg_onto_bg(
    foreground: Image.Image,
    background: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    bg = background.copy()
    alpha = foreground.split()[3]  # extract alpha channel

    # Center the foreground
    x = (bg.width  - foreground.width)  // 2
    y = (bg.height - foreground.height) // 2

    # Paste RGBA image using its alpha
    bg.paste(foreground, (x, y), foreground)

    # Create and paste alpha mask in same location
    mask_on_bg = Image.new("L", bg.size, 0)
    mask_on_bg.paste(alpha, (x, y))

    return bg, mask_on_bg

def remove_background_info(caption: str) -> str:
    caption_split = caption.split(".")
    cleaned_caption = []
    for split in caption_split:
        if 'background' not in split.lower():
            cleaned_caption.append(split)
    cleaned_caption = ".".join(cleaned_caption)
    return cleaned_caption

def apply_artifacts(image: Image.Image) -> Image.Image:
    img = image

    if random.random() < 0.4:  # 60% chance
        np_img = np.array(img).astype(np.float32)
        noise_strength = random.uniform(2, 8)  # subtle
        noise = np.random.normal(0, noise_strength, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img, "RGB")

    if random.random() < 0.5:  # 50% chance
        blur_radius = random.uniform(0.2, 1.2)  # subtle blur
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if random.random() < 0.4:  # 40% chance
        buf = io.BytesIO()
        quality = random.randint(80, 96)  # mild artifacts only
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img    

def generate_background(
    width: int = 256,
    height: int = 256,
    mode: str | None = None,
    foreground_image: Image.Image | None = None,
) -> Image.Image:
    """
    Generate a random background image suitable for graphic design use.

    Modes:
      - 'uniform': uniform grey tones
      - 'gradient': smooth gradient between harmonized colors
      - 'green': green background (like a green screen)
      - 'high_contrast': high contrast background
      - 'hard_negative': hard negative background
      - 'synthetic': synthetic background
    """
    if mode is None:
        sampling_weights = [1/3, 1/3, 1/3]
        mode = random.choices(["uniform", "high_contrast", "synthetic"], weights=sampling_weights)[0]

    # Utility: convert HSL → RGB [0-255]
    def hsl_to_rgb_tuple(h, s, l):
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return tuple(int(x * 255) for x in (r, g, b))

    # --- DARK/LIGHT/GREY BACKGROUND ---
    if mode == "uniform":
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, :] = random.randint(0, 255)
        img = apply_artifacts(Image.fromarray(img, "RGB"))
        return img

    # --- GRADIENT (with hue harmony) ---
    elif mode == "gradient":
        # Pick a base hue and offset second hue slightly for harmony
        base_hue = random.random()
        delta_hue = random.uniform(-0.08, 0.08)
        start_color = np.array(hsl_to_rgb_tuple(
            (base_hue + delta_hue) % 1.0,
            random.uniform(0.2, 0.6),
            random.uniform(0.25, 0.9)
        ), dtype=float)
        end_color = np.array(hsl_to_rgb_tuple(
            (base_hue - delta_hue) % 1.0,
            random.uniform(0.2, 0.6),
            random.uniform(0.25, 0.9)
        ), dtype=float)

        # Random gradient angle
        angle = random.uniform(0, 2 * np.pi)
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y)
        grad = (xv * np.cos(angle) + yv * np.sin(angle))
        grad -= grad.min()
        grad /= grad.max()

        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            gradient[:, :, c] = (start_color[c] * (1 - grad) + end_color[c] * grad)
        gradient = apply_artifacts(Image.fromarray(gradient, "RGB"))
        return gradient

    # --- GREEN BACKGROUND ---
    elif mode == "green":
        # add a slight blur/noise to the image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, :] = (0, 255, 0)
        img = apply_artifacts(Image.fromarray(img, "RGB"))
        return img

    # --- GREEN BACKGROUND ---
    elif mode == "high_contrast":
        # add a slight blur/noise to the image
        # RGB cube corners excluding black and white: R, G, B, Yellow, Magenta, Cyan
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, :] = random.choice(colors)
        img = apply_artifacts(Image.fromarray(img, "RGB"))
        return img

    # --- HARD NEGATIVE BACKGROUND ---
    elif mode == "hard_negative":
        image = foreground_image.convert("RGBA")

        array_image = np.array(image)
        array_foreground = array_image[:, :, :3].astype(np.float32)
        array_mask = (array_image[:, :, 3:] / 255).astype(np.float32)

        foreground_pixel_number = np.sum(array_mask > 0)
        color_foreground_mean = np.mean(array_foreground * array_mask, axis=(0, 1)) \
            * (np.prod(array_foreground.shape[:2]) / foreground_pixel_number)

        # randomly brighten or darken up to 20%
        color_up_or_down = random.choice((-1, 1))
        if color_up_or_down == 1:
            color_foreground_mean += (255 - color_foreground_mean) * random.random() * 0.2
        else:
            color_foreground_mean -= color_foreground_mean * random.random() * 0.2

        color = np.clip(color_foreground_mean, 0, 255).astype(np.uint8)
        img = np.ones((height, width, 3), dtype=np.uint8) * color
        img = apply_artifacts(Image.fromarray(img, "RGB"))

        return img

    # --- SYNTHETIC BACKGROUND ---
    elif mode == "synthetic":
        # path to synthetic background images
        path = "/home/gianfavero/projects/background-removal/data/background_gens"

        # get a random image from the path
        random_image = random.choice(os.listdir(path))

        # open the image
        try:
            image = Image.open(os.path.join(path, random_image))
        except Exception as e:
            print(f"Error opening image: {e}")
            return generate_background(width=width, height=height, mode='uniform')

        # resize the image to the desired width and height
        image = image.resize((width, height))

        return image
    else:
        raise ValueError(f"Unknown background mode: {mode}")

## CPU version refinement
def FB_blur_fusion_foreground_estimator_cpu(image, FG, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FGA = cv2.blur(FG * alpha, (r, r))
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    FG = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG = np.clip(FG, 0, 1)
    return FG, blurred_B

def FB_blur_fusion_foreground_estimator_cpu_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    FG, blur_B = FB_blur_fusion_foreground_estimator_cpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_cpu(image, FG, blur_B, alpha, r=6)[0]

def fast_foreground_estimation(image, mask, r=90) -> Image.Image:
    image = np.array(image, dtype=np.float32) / 255.0
    mask = np.array(mask, dtype=np.float32) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_cpu_2(image, mask, r=r)
    estimated_foreground = (estimated_foreground * 255.0).astype(np.uint8)

    estimated_foreground = Image.fromarray(np.ascontiguousarray(estimated_foreground))

    return estimated_foreground

def revert_alpha_blend(composited_rgb, alpha, bg_color=(0,255,0)):
    """
    Recover original RGBA after compositing on a solid background.
    
    composited_rgb: PIL RGB image of the pasted result
    alpha: PIL grayscale image containing the alpha channel (0–255)
    bg_color: background (R,G,B) tuple used during compositing
    """
    C = np.array(composited_rgb).astype(np.float32)
    A = np.array(alpha).astype(np.float32) / 255.0

    B = np.array(bg_color, dtype=np.float32)[None, None, :]

    F = np.zeros_like(C)

    mask = A > 0
    F[mask] = (C[mask] - B * (1 - A[mask,None])) / A[mask,None]

    # clamp
    F = np.clip(F, 0, 255)

    # reassemble RGBA
    out = np.dstack([F, A * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")