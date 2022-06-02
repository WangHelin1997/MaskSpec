import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.lr_decay as lrd
import utils.misc as misc
from utils.pos_embed import interpolate_pos_embed
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import models.models_vit as models_vit
import models.models_swinTrans as models_swinTrans

from dcase19.engine_run import train_one_epoch, evaluate, evaluate_ensemble
from dcase19.dataset import get_training_set, get_test_set


def get_args_parser():
    parser = argparse.ArgumentParser('dcase19 classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print_freq', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='vit', type=str, metavar='MODEL',
                        help='Type of model to train (vit or swin)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path_test_left', default='./dcase19/data/dcase19_testing_mp3.hdf', type=str,
                        help='test dataset path')
    parser.add_argument('--data_path_test_right', default='./dcase19/data/dcase19_testing_mp3.hdf', type=str,
                        help='test dataset path')
    parser.add_argument('--data_path_test_mid', default='./dcase19/data/dcase19_testing_mp3.hdf', type=str,
                        help='test dataset path') 
    parser.add_argument('--norm_file_left', default='./dcase19/mean_std_128.npy', type=str,
                        help='norm file path')
    parser.add_argument('--norm_file_right', default='./dcase19/mean_std_128.npy', type=str,
                        help='norm file path')
    parser.add_argument('--norm_file_mid', default='./dcase19/mean_std_128.npy', type=str,
                        help='norm file path')
                        
    parser.add_argument('--sample_rate', default=32000, type=int)
    parser.add_argument('--clip_length', default=10, type=int)
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('--in_mem', default=False, type=bool)
    parser.add_argument('--extra_augment', default=True, type=bool)
    parser.add_argument('--roll', default=True, type=bool)
    parser.add_argument('--wavmix', default=True, type=bool)
    parser.add_argument('--specmix', default=True, type=bool)
    parser.add_argument('--mixup_alpha', default=0.2, type=float)

    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--u_patchout', default=200, type=int,
                        help='number of masked patches')
    parser.add_argument('--target_size', default=(128,1000), type=tuple,
                        help='target size')

    parser.add_argument('--output_dir', default='./dcase19/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../dcase19/log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume_left', default='',
                        help='resume of left channel from checkpoint')
    parser.add_argument('--resume_right', default='',
                        help='resume of right channel from checkpoint')
    parser.add_argument('--resume_mid', default='',
                        help='resume of mid channel from checkpoint')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size() # set datasets and dataloaders
    global_rank = misc.get_rank()

    dataset_test_left = get_test_set(
        eval_hdf5=args.data_path_test_left, 
        sample_rate=args.sample_rate, 
        clip_length=args.clip_length)
    sampler_test_left = torch.utils.data.SequentialSampler(dataset_test_left)
    print("Sampler_test = %s" % str(sampler_test_left))
    dataset_test_right = get_test_set(
        eval_hdf5=args.data_path_test_right, 
        sample_rate=args.sample_rate, 
        clip_length=args.clip_length)
    sampler_test_right = torch.utils.data.SequentialSampler(dataset_test_right)
    print("Sampler_test = %s" % str(sampler_test_right))
    dataset_test_mid = get_test_set(
        eval_hdf5=args.data_path_test_mid, 
        sample_rate=args.sample_rate, 
        clip_length=args.clip_length)
    sampler_test_mid = torch.utils.data.SequentialSampler(dataset_test_mid)
    print("Sampler_test = %s" % str(sampler_test_mid))
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_test_left = torch.utils.data.DataLoader(
        dataset_test_left, sampler=sampler_test_left,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_test_right = torch.utils.data.DataLoader(
        dataset_test_right, sampler=sampler_test_right,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_test_mid = torch.utils.data.DataLoader(
        dataset_test_mid, sampler=sampler_test_mid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    assert args.model_type == 'vit' or args.model_type == 'swin', "Only support vit and swin models now."
    if args.model_type == 'vit':
        model_left = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            norm_file=args.norm_file_left,
            u_patchout=args.u_patchout,
            target_size=args.target_size
        )
        model_right = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            norm_file=args.norm_file_right,
            u_patchout=args.u_patchout,
            target_size=args.target_size
        )
        model_mid = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            norm_file=args.norm_file_mid,
            u_patchout=args.u_patchout,
            target_size=args.target_size
        )
    elif args.model_type == 'swin':
        model = models_swinTrans.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            norm_file=args.norm_file
        )

    model_left.to(device)
    model_right.to(device)
    model_mid.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_left, args, args.weight_decay,
        no_weight_decay_list=model_left.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.specmix or args.wavmix:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    args.resume = args.resume_left
    misc.load_model(args=args, model_without_ddp=model_left, optimizer=optimizer, loss_scaler=loss_scaler)
    args.resume = args.resume_right
    misc.load_model(args=args, model_without_ddp=model_right, optimizer=optimizer, loss_scaler=loss_scaler)
    args.resume = args.resume_mid
    misc.load_model(args=args, model_without_ddp=model_mid, optimizer=optimizer, loss_scaler=loss_scaler)

    test_stats = evaluate_ensemble(data_loader_test_left, data_loader_test_right, data_loader_test_mid, model_left, model_right, model_mid, device)
    print(f"Accuracy of the network on the {len(dataset_test_left)} test images: {test_stats['acc1']:.1f}%")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
