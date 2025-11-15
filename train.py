import os
import gc
import glob
import datetime
from contextlib import nullcontext
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Silence TensorFlow INFO and WARNING messages (including "End of sequence" logs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import Config
from loss import PixLoss, ClsLoss
from models.birefnet import BiRefNet
from utils import Logger, AverageMeter, set_seed, check_state_dict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import tensorflow as tf

# Ideogram
from ideogram_dataset import SimpleTrainDataset, ideogram_transform, ideogram_collate_fn
import threading, queue
from concurrent.futures import ThreadPoolExecutor
BUCKET_BATCH_SIZE = 50    # how many TFRecords to group at once
MAX_PREFETCH = 50         # how many dataloaders to keep in memory ahead

parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--ckpt_dir', default='ckpts/tmp', help='Temporary folder')
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
args = parser.parse_args()

config = Config()

if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = config.mixed_precision
    kwargs_handlers = [
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=False),
            utils.GradScalerKwargs(backoff_factor=0.5),
    ]
    if mixed_precision == 'fp8':
        kwargs_handlers.append(utils.AORecipeKwargs())
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=kwargs_handlers,
    )
    accelerator.print(accelerator.state)
    accelerator.print('backbone:', config.bb, ', freeze_bb:', config.freeze_bb)
    args.dist = False

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    if args.use_accelerate:
        device = accelerator.local_process_index
    else:
        device = config.device

if config.rand_seed:
    set_seed(config.rand_seed + device)

steps_st = 0
# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

if args.use_accelerate and accelerator.is_main_process:
    logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
    logger.info("Other hyperparameters:"); logger.info(args)
    print('batch size:', config.batch_size)

######### TFRecord streaming utils #########
def bucket_batches(buckets, size):
    """Yield successive groups of N bucket paths."""
    for i in range(0, len(buckets), size):
        yield buckets[i:i+size]

def prepare_dataloader_from_buckets(bucket_group, batch_size, to_be_distributed=False, val_size=0.1):
    """Load a small group of buckets into a dataloader pair."""
    datasets = [
        SimpleTrainDataset(simple_train_name=bucket, keys=["raw_bytes"], transform=ideogram_transform)
        for bucket in bucket_group
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)

    train_size = int(len(dataset) * (1 - val_size))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.rand_seed)
    )

    # Build loaders
    def make_loader(ds, sampler):
        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=min(config.num_workers, batch_size),
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            collate_fn=ideogram_collate_fn,
        )

    train_sampler = DistributedSampler(train_dataset) if to_be_distributed else None
    val_sampler = DistributedSampler(val_dataset) if to_be_distributed else None
    train_loader = make_loader(train_dataset, train_sampler)
    val_loader = make_loader(val_dataset, val_sampler)
    return train_loader, val_loader


def prefetch_dataloaders(buckets, q, batch_size, to_be_distributed, repeat=False):
    """Background thread: prepare dataloaders and put them into queue."""
    while True:
        for bucket_group in bucket_batches(buckets, BUCKET_BATCH_SIZE):
            loaders = prepare_dataloader_from_buckets(bucket_group, batch_size, to_be_distributed)
            q.put(loaders)  # blocks if queue is full (maxsize=MAX_PREFETCH)
        if not repeat:
            break
    q.put(None)  # signal end


def stream_ideogram_loaders(buckets, batch_size, to_be_distributed, repeat=False):
    """Background thread: prepare dataloaders and put them into queue."""
    q = queue.Queue(maxsize=MAX_PREFETCH)
    t = threading.Thread(
        target=prefetch_dataloaders,
        args=(buckets, q, batch_size, to_be_distributed, repeat),
        daemon=True,
    )
    t.start()

    while True:
        loaders = q.get()
        if loaders is None:
            break
        yield loaders

def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=False and not os.path.isfile(str(args.resume)))
    else:
        print('Undefined model: {}.'.format(config.model))
        return None
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global steps_st
            try:
                steps_st = int(args.resume.rstrip('.pth').split('step_')[-1]) + 1
            except:
                steps_st = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=1e-4)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=0)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
    #     gamma=config.lr_decay_rate
    # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,          # full run length
        eta_min=config.lr * 0.01  # don’t go fully to zero
    )
    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, train_loader, val_loader, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader = train_loader
        self.val_loader = val_loader

        if args.use_accelerate:
            self.train_loader, self.val_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.val_loader, self.model, self.optimizer)
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()
        
        # Others
        self.loss_log = AverageMeter()

    def _test_batch(self):
        batch = next(iter(self.train_loader))
        inputs = batch[0]
        gts = batch[1]
        class_labels = batch[2]
        return inputs, gts, class_labels

    def _train_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
        self.optimizer.zero_grad()
        scaled_preds, class_preds_lst = self.model(inputs)
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            # self.loss_dict['loss_gdt'] = loss_gdt.item()
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels)
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        self.loss_log.update(loss.item(), inputs.size(0))
        if args.use_accelerate:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()

    def _validate_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
        scaled_preds, class_preds_lst = self.model(inputs)
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            # self.loss_dict['loss_gdt'] = loss_gdt.item()
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels)
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        self.loss_log.update(loss.item(), inputs.size(0))

    def train_epoch(self, steps):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}
        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % max(100, len(self.train_loader) / 100 // 100 * 100) == 0:
                info_progress = f'Step[{steps + batch_idx}].'
                info_loss = 'Training Losses:'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += f' {loss_name}: {loss_value:.5g} |'
                if args.use_accelerate and accelerator.is_main_process:
                    logger.info(' '.join((info_progress, info_loss)))
        info_loss = f'@==Final== Step[{steps + batch_idx}]  Training Loss: {self.loss_log.avg:.5g}  '
        if args.use_accelerate and accelerator.is_main_process:
            logger.info(info_loss)

        self.lr_scheduler.step()
        return steps + batch_idx + 1, self.loss_log.avg

    @torch.no_grad()
    def validate_epoch(self, steps):
        global logger_loss_idx
        self.loss_dict = {}
        self.loss_log.reset()

        for batch_idx, batch in enumerate(self.val_loader):
            self._validate_batch(batch)
            # Logger
            if batch_idx % max(100, len(self.val_loader) / 100 // 100 * 100) == 0:
                info_progress = f'Step[{steps}].'
                info_loss = 'Validation Losses:'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += f' {loss_name}: {loss_value:.5g} |'
                if args.use_accelerate and accelerator.is_main_process:
                    logger.info(' '.join((info_progress, info_loss)))
        info_loss = f'@==Final== Step[{steps}]  Validation Loss: {self.loss_log.avg:.5g}  '
        if args.use_accelerate and accelerator.is_main_process:
            logger.info(info_loss)

        return self.loss_log.avg

def main():
    # Put Ideogram GCP data buckets here
    buckets = [f"gs://ideogram-data-snapshots-us-east5/loras/rgba100k/mp.tfr-{i:05d}" for i in range(0, 512)]
    buckets.extend([
        "gs://ideogram-data-snapshots-us-east5/loras/rgba100k/tee_logo_1k.tfr",
    ])

    # Streaming dataloaders
    stream = stream_ideogram_loaders(buckets, config.batch_size, to_be_distributed, repeat=True)

    # Initialize models and optimizers
    model_opt_lrsch = init_models_optimizers(args.epochs, to_be_distributed)

    # Train
    steps = steps_st
    for i, (train_loader, val_loader) in enumerate(stream):
        
        trainer = Trainer(train_loader, val_loader, model_opt_lrsch)
        
        steps, train_loss = trainer.train_epoch(steps)

        if i % 5 == 0:
            val_loss = trainer.validate_epoch(steps)

        if i % 10 == 0:
            if args.use_accelerate:
                state_dict = trainer.model.state_dict()
            else:
                state_dict = trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()
            torch.save(state_dict, os.path.join(args.ckpt_dir, 'step_{}.pth'.format(steps)))

        # free memory before moving to next batch
        del trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()