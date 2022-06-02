import argparse
from ast import arg
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
import timm.optim.optim_factory as optim_factory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import models.models_mae as models_mae
import models.models_simMIM as models_simMIM

from trainer.engine_pretrain import train_one_epoch
from audioset.dataset import get_full_training_set, get_random_sampler, get_base_training_set, get_other_sets

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--print_freq', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='vit', type=str, metavar='MODEL',
                        help='Type of model to train (vit or swin)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.8, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--balanced_train_hdf5', default='/data/dean/whl/audioset_Kong/mp3/balanced_train_segments_mp3.hdf', type=str,
                        help='balanced train dataset path')
    parser.add_argument('--unbalanced_train_hdf5', default='/data/dean/whl/audioset_Kong/mp3/unbalanced_train_segments_mp3.hdf', type=str,
                        help='unbalanced train dataset path')
    parser.add_argument('--eval_hdf5', default='/data/dean/whl/audioset_Kong/mp3/eval_segments_mp3.hdf', type=str,
                        help='eval dataset path')
    parser.add_argument('--other_hdf5_path', default='', type=str,
                        help='other datasets path')
    parser.add_argument('--use_audioset', default=True, type=bool)
    parser.add_argument('--norm_file', default='./audioset/mean_std_128.npy', type=str,
                        help='norm file path')
    parser.add_argument('--sample_rate', default=32000, type=int)
    parser.add_argument('--clip_length', default=10, type=int)
    parser.add_argument('--augment', default=False, type=bool)
    parser.add_argument('--in_mem', default=False, type=bool)
    parser.add_argument('--extra_augment', default=False, type=bool)
    parser.add_argument('--roll', default=False, type=bool)
    parser.add_argument('--wavmix', default=False, type=bool)
    parser.add_argument('--random_sample', default=False, type=bool)
    parser.add_argument('--epoch_len', default=100000, type=int)
    parser.add_argument('--only_balanced', action='store_true',
                        help='Use balanced audioset for pretrain (debug)')
    parser.set_defaults(only_balanced=False)
    parser.add_argument('--use_othersets', action='store_true',
                        help='Use other datasets for pretrain')
    parser.set_defaults(use_othersets=False)

    parser.add_argument('--output_dir', default='./output_dir_pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_dir', default='',
                        help='resume dir')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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
    print("Seed: {}".format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.resume_dir and not args.resume:
        tag = ''
        for root, dirs, files in os.walk(args.resume_dir, topdown=False):
            for name in files:
                if name[-3:] == 'pth':
                    if not tag:
                        tag = os.path.join(root, name)
                    elif int(name.split('checkpoint-')[1].split('.pth')[0]) > int(tag.split('checkpoint-')[1].split('.pth')[0]):
                        tag = os.path.join(root, name)
        args.resume = tag

    cudnn.benchmark = True

    if args.only_balanced:
        dataset_train = get_base_training_set(
                balanced_train_hdf5=args.balanced_train_hdf5, 
                sample_rate=args.sample_rate, 
                clip_length=args.clip_length, 
                augment=args.augment, 
                in_mem=args.in_mem, 
                extra_augment=args.extra_augment, 
                roll=args.roll,
                wavmix=args.wavmix)
    elif args.use_othersets:
        dataset_train = get_other_sets(
            others_hdf5_path=args.others_hdf5_path, 
            use_audioset=args.use_audioset,
            balanced_train_hdf5=args.balanced_train_hdf5, 
            unbalanced_train_hdf5=args.unbalanced_train_hdf5,
            sample_rate=args.sample_rate, 
            clip_length=args.clip_length, 
            augment=args.augment, 
            in_mem=args.in_mem, 
            extra_augment=args.extra_augment, 
            roll=args.roll, 
            wavmix=args.wavmix)
    else:
        dataset_train = get_full_training_set(
            balanced_train_hdf5=args.balanced_train_hdf5, 
            unbalanced_train_hdf5=args.unbalanced_train_hdf5,
            sample_rate=args.sample_rate, 
            clip_length=args.clip_length, 
            augment=args.augment, 
            in_mem=args.in_mem, 
            extra_augment=args.extra_augment, 
            roll=args.roll, 
            wavmix=args.wavmix)
    print(dataset_train)

    if args.random_sample and not args.only_balanced:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = get_random_sampler(dataset_train, epoch_len=args.epoch_len)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    assert args.model_type == 'vit' or args.model_type == 'swin', "Only support vit and swin models now."
    # define the model
    if args.model_type == 'vit':
        model = models_mae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            norm_file=args.norm_file
            )
    elif args.model_type == 'swin':
        model = models_simMIM.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            norm_file=args.norm_file
            )

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
