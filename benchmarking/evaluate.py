'''
Evaluate a given model on a given benchmark

Example:
python evaluate.py --model_name birefnet --benchmark gcp_url_to_benchmark

Gian Favero
Ideogram
2025-10-29
'''

import sys
sys.path.insert(0, "/home/gianfavero/projects/")
sys.path.insert(0, "/home/gianfavero/projects/BiRefNet/")

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from BiRefNet.benchmarking.factory import get_model
from BiRefNet.ideogram_dataset import BenchmarkDataset, EvalDataset
from BiRefNet.ideogram_utils import pil_image_to_bytes, reduce_spill, recover_original_rgba
from tfrecords.benchmark.tfr import BenchmarkExample
from tfrecords.eval.tfr import EvalExample

from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

##### Functional code for evaluation #####

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

@torch.no_grad()
def evaluate(model, dataloader):
    torch.set_float32_matmul_precision(['high', 'highest'][0])

    images_list = []
    masks_list = []
    for batch in dataloader:
        input_images = batch["input_images"].to(model.device).half() # needs to be full precision for rmbgv2
        images = batch["images"]

        masks = model(input_images)
        masks[masks < 0.1] = 0

        images_list.extend(images)
        masks_list.append(masks.detach().cpu())
    masks_list = torch.cat(masks_list, dim=0)

    output_list = []
    for image, mask in zip(images_list, masks_list):
        mask = transforms.ToPILImage()(mask)
        mask = mask.resize(image.size)

        #recovered_rgba = recover_original_rgba(image, mask)
        image = reduce_spill(image, mask, r=90)

        image.putalpha(mask)

        output_list.append(image)

    return output_list

def save_output(output_list, model_name, benchmark):
    os.makedirs(f"eval-output/{benchmark}/{model_name}", exist_ok=True)
    for i, output in enumerate(output_list):
        output.save(f"eval-output/{benchmark}/{model_name}/sample_{i}.png")

def write_output_to_tfr(output_list, benchmark_url, writer_name):
    data = tf.data.TFRecordDataset(str(benchmark_url)).as_numpy_iterator()
    with tf.io.TFRecordWriter(writer_name) as writer:
        for image, serialized in zip(output_list, data):
            ex = EvalExample.from_tf_example(serialized)
            writer.write(
                EvalExample(
                    prompt=ex.prompt,
                    images=[pil_image_to_bytes(image)]
                ).to_tf_example().SerializeToString()
            )
    print(f"Wrote {len(output_list)} eval examples to {writer_name}")

def main(model_name, benchmark, device, path_to_weight):
    print(f"Evaluating {model_name} on {benchmark}")
    
    if benchmark == "green-benchmark":
        benchmark_url = "gs://mobius-dev-us-east5/gian_favero_workspace/background_removal_examples/11122025_bg_removal_bm_gian-green-graphic-2k.tfr"
    elif benchmark == "base-benchmark":
        benchmark_url = "gs://mobius-dev-us-east5/gian_favero_workspace/background_removal_examples/11122025_bg_removal_bm_v3.tfr"
    elif benchmark == "ig-benchmark":
        benchmark_url = "gs://mobius-dev-us-east5/gian_favero_workspace/background_removal_examples/10292025_samples.tfr"

    tfr_dataset = EvalDataset( 
        writer_name=benchmark_url,
        keys=["images"],
        transform=bg_removal_transform,
    )

    tfr_dataloader = DataLoader(
        tfr_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_model(model_name, device=device, path_to_weight=path_to_weight)

    output = evaluate(model, tfr_dataloader)

    writer_name = f"gs://mobius-dev-us-east5/gian_favero_workspace/background_removal_examples/removed_{benchmark_url.split('/')[-1]}"
    #write_output_to_tfr(output, benchmark_url, writer_name)

    save_output(output, model_name, benchmark)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path_to_weight", type=str, default="None")
    args = parser.parse_args()

    main(args.model_name, args.benchmark, args.device, args.path_to_weight)