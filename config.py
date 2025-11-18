import os
import math


class Config():
    def __init__(self) -> None:
        # Main active settings
        self.batch_size = 4                                     # Multi-GPU+BF16 training for 76GB / 62GB, without/with compile, on each A100.
        self.compile = False # turn on after                    # 1. PyTorch<=2.0.1 has an inherent CPU memory leak problem; 2.0.1<PyTorch<2.5.0 cannot successfully compile.
        self.mixed_precision = ['no', 'fp16', 'bf16', 'fp8'][1] # 2. FP8 doesn't show acceleration in the torch.compile mode.
        self.SDPA_enabled = True                                # H200x1 + compile==True.  None: 43GB + 14s, math: 43GB + 15s, mem_eff: 35GB + 15s.
                                                                # H200x1 + compile==False. None: 54GB + 25s, math: 51GB + 26s, mem_eff: 40GB + 25s.

        # TASK settings
        self.task = 'Matting'

        # Faster-Training settings
        self.precisionHigh = True
       
        # MODEL settings
        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = [0, 3][1]    # multi-scale skip connections from encoder
        self.mul_scl_ipt = ['', 'add', 'cat'][2]
        self.dec_att = ['', 'ASPP', 'ASPPDeformable'][2]
        self.squeeze_block = ['', 'BasicDecBlk_x1', 'ResBlk_x4', 'ASPP_x3', 'ASPPDeformable_x3'][1]
        self.dec_blk = ['BasicDecBlk', 'ResBlk'][0]

        # TRAINING settings
        self.lr = 1e-5 #(1e-4 if 'DIS5K' in self.task else 1e-5) * math.sqrt(self.batch_size / 4)     # DIS needs high lr to converge faster. Adapt the lr linearly
        self.num_workers = 1 #max(4, self.batch_size)          # will be decreased to min(it, batch_size) at the initialization of the data_loader

        # Backbone settings
        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',

            'swin_v1_l', 'swin_v1_b',
            'swin_v1_s', 'swin_v1_t',

            'pvt_v2_b5', 'pvt_v2_b2',
            'pvt_v2_b1', 'pvt_v2_b0',

            'dino_v3_7b', 'dino_v3_h_plus', 'dino_v3_l',
            'dino_v3_b', 'dino_v3_s_plus', 'dino_v3_s',
        ][3]
        self.freeze_bb = 'dino_v3' in self.bb
        self.lateral_channels_in_collection = {
            'vgg16': [512, 512, 256, 128], 'vgg16bn': [512, 512, 256, 128], 'resnet50': [2048, 1024, 512, 256],

            'dino_v3_7b': [4096] * 4, 'dino_v3_h_plus': [1280] * 4, 'dino_v3_l': [1024] * 4,
            'dino_v3_b': [768] * 4, 'dino_v3_s_plus': [384] * 4, 'dino_v3_s': [384] * 4,

            'swin_v1_l': [1536, 768, 384, 192], 'swin_v1_b': [1024, 512, 256, 128],
            'swin_v1_s': [768, 384, 192, 96], 'swin_v1_t': [768, 384, 192, 96],

            'pvt_v2_b5': [512, 320, 128, 64], 'pvt_v2_b2': [512, 320, 128, 64],
            'pvt_v2_b1': [512, 320, 128, 64], 'pvt_v2_b0': [256, 160, 64, 32],
        }[self.bb]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        # MODEL settings - inactive
        self.lat_blk = ['BasicLatBlk'][0]
        self.dec_channels_inter = ['fixed', 'adap'][0]
        self.auxiliary_classification = False       # Only for DIS5K, where class labels are saved in `dataset.py`.
        self.model = [
            'BiRefNet',
        ][0]

        # TRAINING settings - inactive
        self.optimizer = ['Adam', 'AdamW'][1]
        self.lr_decay_epochs = [1e5]    # Set to negative N to decay the lr in the last N-th epoch.
        self.lr_decay_rate = 0.5
        # Loss
        if self.task in ['Matting']:
            self.lambdas_pix_last = {
                'bce': 10 * 1,
                'iou': 0.5 * 0,
                'iou_patch': 0.5 * 0,
                'mae': 100 * 1,
                'mse': 30 * 0,
                'triplet': 3 * 0,
                'reg': 100 * 0,
                'ssim': 10 * 1,
                'cnt': 5 * 0,
                'structure': 5 * 0,
            }
        elif self.task in ['Custom', 'General-2K']:
            self.lambdas_pix_last = {
                'bce': 30 * 1,
                'iou': 0.5 * 1,
                'iou_patch': 0.5 * 0,
                'mae': 100 * 1,
                'mse': 30 * 0,
                'triplet': 3 * 0,
                'reg': 100 * 0,
                'ssim': 10 * 1,
                'cnt': 5 * 0,
                'structure': 5 * 0,
            }
        else:
            self.lambdas_pix_last = {
                # not 0 means opening this loss
                # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
                'bce': 30 * 1,          # high performance
                'iou': 0.5 * 1,         # 0 / 255
                'iou_patch': 0.5 * 0,   # 0 / 255, win_size = (64, 64)
                'mae': 30 * 0,
                'mse': 30 * 0,         # can smooth the saliency map
                'triplet': 3 * 0,
                'reg': 100 * 0,
                'ssim': 10 * 1,          # help contours,
                'cnt': 5 * 0,          # help contours
                'structure': 5 * 0,    # structure loss from codes of MVANet. A little improvement on DIS-TE[1,2,3], a bit more decrease on DIS-TE4.
            }
        self.lambdas_cls = {
            'ce': 5.0
        }

        # Callbacks - inactive
        self.verbose_eval = True
        self.only_S_MAE = False

        # others
        self.device = [0, 'cpu'][0]     # .to(0) == .to('cuda:0')

        self.batch_size_valid = 1
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'train.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'train.sh' == f]
        if run_sh_file:
            with open(run_sh_file[0], 'r') as f:
                lines = f.readlines()
                self.save_last = int([l.strip() for l in lines if "'{}')".format(self.task) in l and 'val_last=' in l][0].split('val_last=')[-1].split()[0])
                self.save_step = int([l.strip() for l in lines if "'{}')".format(self.task) in l and 'step=' in l][0].split('step=')[-1].split()[0])


# Return task for choosing settings in shell scripts.
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Only choose one argument to activate.')
    parser.add_argument('--print_task', action='store_true', help='print task name')
    parser.add_argument('--print_testsets', action='store_true', help='print validation set')
    args = parser.parse_args()

    config = Config()
    for arg_name, arg_value in args._get_kwargs():
        if arg_value:
            print(config.__getattribute__(arg_name[len('print_'):]))

