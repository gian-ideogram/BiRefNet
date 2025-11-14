'''
Loading data from GCP.

Gian Favero
Ideogram
2025-10-29
'''

import sys
sys.path.insert(0, "/home/gianfavero/projects/tfrecords")

import PIL
import io
from io import BytesIO
from typing import TypedDict
import random

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from tfrecords.simple_train.tfr import SimpleTrain
from tfrecords.benchmark.tfr import BenchmarkExample
from tfrecords.eval.tfr import EvalExample
from .ideogram_utils import generate_background, paste_fg_onto_bg, crop_to_visible_area, resize_image_for_square

class ColourPalette(TypedDict):
    palette: bytes
    weight: bytes

simple_train_field_types = {
    "raw_bytes": bytes,
    "raw_bytes_sha256": bytes,
    "original_raw_bytes_sha256": bytes,
    "captions": dict[str, str],
    "classifier_scores": dict[str, float],
    "colour_palettes": dict[str, ColourPalette],
}

aspect_ratios = {
    "1:1": (1024, 1024),
    "2:3": (832, 1248),
    "3:2": (1248, 832),
    "3:4": (864, 1152),
    "4:3": (1152, 864),
    "4:5": (896, 1120),
    "5:4": (1120, 896),
    "16:10": (1280, 800),
    "10:16": (800, 1280),
}

def ideogram_transform(sample):
    # Assume the input is an RGBA image
    rgba_image = PIL.Image.open(BytesIO(sample["raw_bytes"]))
    rgba_image = rgba_image.convert("RGBA")
    rgba_image = crop_to_visible_area(rgba_image)
    rgba_image = resize_image_for_square(rgba_image, h_w=1024, fraction=0.8)

    # Pick a random aspect ratio
    aspect_ratio = random.choice(list(aspect_ratios.keys()))
    width, height = aspect_ratios[aspect_ratio]

    # Generate a green background with a random aspect ratio
    background = generate_background(width=width, height=height, mode=None, foreground_image=rgba_image)
    rgba_image_with_bg, alpha_mask = paste_fg_onto_bg(rgba_image, background)

    input_transform_pipeline = transforms.Compose([
        transforms.Resize((1024, 1024)), # fixed size in BiRefNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform_pipeline = transforms.Compose([
        transforms.Resize((1024, 1024)), # fixed size in BiRefNet
        transforms.ToTensor(),
    ])

    input_image = input_transform_pipeline(rgba_image_with_bg) # Tensor (C, H, W) in the range [0.0, 1.0]
    mask = mask_transform_pipeline(alpha_mask) # Tensor (1, H, W) in the range [0.0, 1.0]
    return input_image, mask

def ideogram_collate_fn(batch):
    input_images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    class_labels = [torch.tensor(-1, dtype=torch.long)] * len(batch) # dummy class label
    input_images = torch.stack(input_images)
    masks = torch.stack(masks)
    class_labels = torch.stack(class_labels)
    return (input_images, masks, class_labels)

class SimpleTrainDataset(Dataset):
    def __init__(self, simple_train_name: str, keys: list[str], transform = None):
        self.simple_train_name = simple_train_name
        self.keys = keys
        self.simple_train = self._get_simple_train()
        self.transform = transform

    def __len__(self):
        return len(self.simple_train)

    def __getitem__(self, idx):
        sample = self.simple_train[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_simple_train(self):
        data = tf.data.TFRecordDataset(
            self.simple_train_name
        ).as_numpy_iterator()

        parsed_data = self._parse_simple_train(data)

        return parsed_data

    def _parse_simple_train(self, data: list[bytes]):
        parsed_examples = []

        for serialized in data:
            ex = SimpleTrain.from_tf_example(serialized) # convert to SimpleTrain
            parsed_data = {}

            for key in self.keys:
                if key not in simple_train_field_types:
                    raise ValueError(f"Unsupported key: {key}")

                field_type = simple_train_field_types[key]
                field_value = getattr(ex, key)

                if field_type is bytes:
                    parsed_data[key] = field_value
                else:
                    parsed_data[key] = field_value

            parsed_examples.append(parsed_data)

        return parsed_examples

bm_field_types = {
    "id": str,
    "text": str,
    "num_samples": int,
    "quoted_text": str,
    "inpainting_image": bytes,
    "inpainting_mask": bytes,
    "colour_palette": bytes,
    "colour_palette_weight": bytes,
    "style_reference_image": bytes,
    "entity_reference_images": list,
    "entity_reference_types": list,
    "entity_reference_metadata": list,
    "images": list,
}

class BenchmarkDataset(Dataset):
    def __init__(self, writer_name: str, keys: list[str], transform = None):
        self.writer_name = writer_name
        self.keys = keys
        self.benchmark = self._get_benchmark()
        self.transform = transform

    def __len__(self):
        return len(self.benchmark)

    def __getitem__(self, idx):
        sample = self.benchmark[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_benchmark(self):
        data = tf.data.TFRecordDataset(
            self.writer_name
        ).as_numpy_iterator()

        parsed_data = self._parse_benchmark(data)

        return parsed_data

    def _parse_benchmark(self, data: list[bytes]):
        parsed_examples = []

        for serialized in data:
            ex = BenchmarkExample.from_tf_example(serialized)
            parsed_data = {}

            for key in self.keys:
                if key not in bm_field_types:
                    raise ValueError(f"Unsupported key: {key}")

                field_type = bm_field_types[key]
                field_value = getattr(ex, key)

                if field_type is bytes:
                    parsed_data[key] = PIL.Image.open(BytesIO(field_value))
                elif field_type is list: # assume list of bytes
                    parsed_data[key] = [PIL.Image.open(BytesIO(item)) for item in field_value]
                else:
                    parsed_data[key] = field_value

            parsed_examples.append(parsed_data)

        return parsed_examples

eval_field_types = {
    "prompt": BenchmarkExample,
    "images": list[bytes],
}

class EvalDataset(Dataset):
    def __init__(self, writer_name: str, keys: list[str], transform = None):
        self.writer_name = writer_name
        self.keys = keys
        self.eval_set = self._get_eval_set()
        self.transform = transform

    def __len__(self):
        return len(self.eval_set)

    def __getitem__(self, idx):
        sample = self.eval_set[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_eval_set(self):
        data = tf.data.TFRecordDataset(
            self.writer_name
        ).as_numpy_iterator()

        parsed_data = self._parse_eval_set(data)

        return parsed_data

    def _parse_eval_set(self, data: list[bytes]):
        parsed_examples = []
        for serialized in data:
            ex = EvalExample.from_tf_example(serialized)
            parsed_data = {}
            for key in self.keys:
                if key not in eval_field_types:
                    raise ValueError(f"Unsupported key: {key}")

                field_type = eval_field_types[key]
                field_value = getattr(ex, key)

                if field_type == bytes:
                    parsed_data[key] = PIL.Image.open(BytesIO(field_value))
                elif field_type == list[bytes]: # assume list of bytes
                    parsed_data[key] = [PIL.Image.open(BytesIO(item)) for item in field_value]
                else:
                    parsed_data[key] = field_value

            parsed_examples.append(parsed_data)

        return parsed_examples