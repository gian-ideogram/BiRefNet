import re
import os
import numpy as np
import math

from torch.utils.tensorboard import SummaryWriter

STEPS_PER_EPOCH = 101

def parse_manual_log(log_path):
    """
    Parse a BiRefNet-style training log file into a structured dictionary:
    {
      epoch_number: {
        'bce': float,
        'iou': float,
        'ssim': float,
        'mae': float,
        'loss_pix': float,
        'final': float
      }
    }
    """
    # Pattern for iteration loss lines
    iter_pattern = re.compile(
        r"Epoch\[(\d+)/\d+\].*Training Losses: bce:\s([\d\.eE+-]+)\s\|\siou:\s([\d\.eE+-]+)\s\|\sssim:\s([\d\.eE+-]+)\s\|\smae:\s([\d\.eE+-]+)\s\|\sloss_pix:\s([\d\.eE+-]+)"
    )

    # Pattern for final summary line
    final_pattern = re.compile(
        r"@==Final==\sEpoch\[(\d+)/\d+\]\s+Training Loss:\s([\d\.eE+-]+)"
    )

    results = {}

    with open(log_path, "r") as f:
        for line in f:
            # Match per-iteration training losses
            iter_match = iter_pattern.search(line)
            if iter_match:
                epoch = int(iter_match.group(1))
                step = (epoch - 244) * STEPS_PER_EPOCH
                # only record first one if we haven't already
                if step not in results:
                    bce, iou, ssim, mae, loss_pix = map(float, iter_match.groups()[1:])
                    results[step] = {
                        "bce": bce,
                        "iou": iou,
                        "ssim": ssim,
                        "mae": mae,
                        "loss_pix": loss_pix,
                    }

            # Match the final epoch loss
            final_match = final_pattern.search(line)
            if final_match:
                epoch = int(final_match.group(1))
                step = (epoch - 244) * STEPS_PER_EPOCH
                # only record first final loss if not already stored
                if "final" not in results.get(step, {}):
                    final_loss = float(final_match.group(2))
                    results.setdefault(step, {})
                    results[step]["final"] = final_loss

    return results

def parse_training_log(log_path):
    """
    Parse a BiRefNet-style training log file into a structured dictionary:
    {
      epoch_number: {
        'bce': float,
        'iou': float,
        'ssim': float,
        'mae': float,
        'loss_pix': float,
        'final': float
      }
    }
    """
    # Regex to match the training loss line
    iter_pattern = re.compile(
        r"Epoch\[(\d+)/\d+\].*Training Losses: bce:\s([\d\.eE+-]+)\s\|\siou:\s([\d\.eE+-]+)\s\|\sssim:\s([\d\.eE+-]+)\s\|\smae:\s([\d\.eE+-]+)\s\|\sloss_pix:\s([\d\.eE+-]+)"
    )

    # Regex to match the final epoch summary line
    final_pattern = re.compile(
        r"@==Final==\sEpoch\[(\d+)/\d+\]\s+Training Loss:\s([\d\.eE+-]+)"
    )

    results = {}

    with open(log_path, "r") as f:
        for line in f:
            # Match the iteration loss line
            iter_match = iter_pattern.search(line)
            if iter_match:
                epoch = int(iter_match.group(1))
                step = (epoch - 244) * STEPS_PER_EPOCH  # steps starts at 0
                bce, iou, ssim, mae, loss_pix = map(float, iter_match.groups()[1:])
                results.setdefault(step, {})  # use step instead of epoch
                results[step].update({
                    "bce": bce,
                    "iou": iou,
                    "ssim": ssim,
                    "mae": mae,
                    "loss_pix": loss_pix,
                })
            
            # Match the final line
            final_match = final_pattern.search(line)
            if final_match:
                epoch = int(final_match.group(1))
                step = (epoch - 244) * STEPS_PER_EPOCH  # final loss after full epoch, assign to end-of-epoch step
                final_loss = float(final_match.group(2))
                results.setdefault(step, {})
                results[step]["final"] = final_loss

    return results

def parse_validation_log(log_path):
    """
    Parse a BiRefNet-style validation log file into a structured dictionary:
    {
      epoch_number: {
        'bce': float,
        'iou': float,
        'ssim': float,
        'mae': float,
        'loss_pix': float,
        'final': float
      }
    }
    """
    # Regex to match the validation loss line
    iter_pattern = re.compile(
            r"Epoch\[(\d+)/\d+\].*Validation Losses: bce:\s([\d\.eE+-]+)\s\|\siou:\s([\d\.eE+-]+)\s\|\sssim:\s([\d\.eE+-]+)\s\|\smae:\s([\d\.eE+-]+)\s\|\sloss_pix:\s([\d\.eE+-]+)"
    )

    # Regex to match the final epoch summary line
    final_pattern = re.compile(
        r"@==Final==\sEpoch\[(\d+)/\d+\]\s+Validation Loss:\s([\d\.eE+-]+)"
    )

    results = {}

    with open(log_path, "r") as f:
        for line in f:
            # Match the iteration loss line
            iter_match = iter_pattern.search(line)
            if iter_match:
                epoch = int(iter_match.group(1))
                step = (epoch - 244) * STEPS_PER_EPOCH  # steps starts at 0
                bce, iou, ssim, mae, loss_pix = map(float, iter_match.groups()[1:])
                results.setdefault(step, {})  # use step instead of epoch
                results[step].update({
                    "bce": bce,
                    "iou": iou,
                    "ssim": ssim,
                    "mae": mae,
                    "loss_pix": loss_pix,
                })
            
            # Match the final line
            final_match = final_pattern.search(line)
            if final_match:
                epoch = int(final_match.group(1))
                step = (epoch - 244) * STEPS_PER_EPOCH  # final loss after full epoch, assign to end-of-epoch step
                final_loss = float(final_match.group(2))
                results.setdefault(step, {})
                results[step]["final"] = final_loss

    return results

def parse_eval_table_to_steps(text):
    """
    Parse an ASCII evaluation table into a dictionary keyed by training steps.

    Args:
        text (str): multiline table text

    Returns:
        dict: {step_number: {metric_name: float, ...}}
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header_line = next((l for l in lines if l.startswith("&") and "Dataset" in l), None)
    if header_line is None:
        raise ValueError("No header found in table")

    headers = [h.strip() for h in header_line.split("&") if h.strip() and not h.startswith("+")]
    data_lines = [l for l in lines if l.startswith("&") and "Dataset" not in l]

    results = {}

    for line in data_lines:
        parts = [p.strip() for p in line.split("&") if p.strip()]
        if len(parts) != len(headers):
            continue

        row = dict(zip(headers, parts))
        match = re.search(r"epoch_(\d+)", row.get("Method", ""))
        if not match:
            continue

        epoch = int(match.group(1))
        step = (epoch - 244) * STEPS_PER_EPOCH

        epoch_data = {}
        for key, val in row.items():
            if key in ["Dataset", "Method"]:
                continue
            val = val.strip()
            if val.lower() == "nan" or val.lower() == ".000":
                continue
            else:
                try:
                    if re.match(r"^\.\d+$", val):
                        val = "0" + val
                    epoch_data[key] = float(val)
                except ValueError:
                    epoch_data[key] = val

        results[step] = epoch_data

    # Sort by step (increasing)
    results = dict(sorted(results.items()))

    return results

def log_dict_to_tensorboard(results, log_dir="runs/from_log", prefix="train"):
    """
    Log parsed losses (nested dict keyed by epoch) to TensorBoard.

    Args:
        results (dict): {epoch: {loss_name: value, ...}}
        log_dir (str): output directory for TensorBoard files
        prefix (str): prefix for scalar names (e.g., "train" or "val")
    """
    writer = SummaryWriter(log_dir)
    for epoch, losses in sorted(results.items()):
        for name, value in losses.items():
            writer.add_scalar(f"{prefix}/{name}", value, epoch)
    writer.close()

def log_hparams_to_tensorboard(hparams, log_dir="runs/from_log"):
    """
    Log hyperparameters to TensorBoard.

    Args:
        hparams (dict): hyperparameters to log
        log_dir (str): output directory for TensorBoard files
    """
    writer = SummaryWriter(log_dir)
    for key, value in hparams.items():
        writer.add_text(key, str(value), 0)
    writer.close()  

if __name__ == "__main__":
    # Define hyperparameters
    hparams = {
        "dataset": "rgba20k_tshirtlogo1k",
        "dynamic_size": True,
        "mixed_precision": "fp16",
        "gpus": 8,
        "batch_size": 4,
        "epochs": 100,
        "steps_per_epoch": 101,
        "steps": 10100,
        "lr": 1e-5,
        "lr_decay_epochs": 1e5,
        "optimizer": "AdamW",
        "lr_decay_rate": 0.5,
    }

    # Define path to log files
    train_log_path = "/home/gianfavero/projects/birefnet-project/codes/dis/BiRefNet/ckpts/varied_1e-5/log.txt" # same log for both

    # Define path to save the TensorBoard logs
    tb_log_dir = "ckpts/tensorboard/varied_1e-5"

    # Parse the training log file
    parsed_log = parse_training_log(train_log_path)

    # Parse the evaluation log file
    parsed_eval_log = parse_validation_log(train_log_path)
    
    # Log the hyperparameters to TensorBoard
    log_hparams_to_tensorboard(hparams, tb_log_dir)

    # Log the parsed log to TensorBoard
    log_dict_to_tensorboard(parsed_log, tb_log_dir, "train")
    log_dict_to_tensorboard(parsed_eval_log, tb_log_dir, "val")