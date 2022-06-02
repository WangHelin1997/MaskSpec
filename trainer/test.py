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
from timm.loss import BinaryCrossEntropy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.lr_decay as lrd
import utils.misc as misc
from utils.pos_embed import interpolate_pos_embed
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import models.models_vit as models_vit
import models.models_swinTrans as models_swinTrans

from trainer.engine_finetune import train_one_epoch, evaluate
from audioset.dataset import get_ft_weighted_sampler, get_full_training_set, get_test_set, get_base_training_set
import librosa
import csv

def get_args_parser():
    parser = argparse.ArgumentParser('Test', add_help=False)
    parser.add_argument('--test_mode', type=str, default='single',
                        help='Test mode (single or mAP)')
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--csv_file', type=str)
    parser.add_argument('--topk', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print_freq', default=2000, type=int)

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
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
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
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--balanced_train_hdf5', default='/data/dean/whl/audioset_Kong/mp3/balanced_train_segments_mp3.hdf', type=str,
                        help='balanced train dataset path')
    parser.add_argument('--unbalanced_train_hdf5', default='/data/dean/whl/audioset_Kong/mp3/unbalanced_train_segments_mp3.hdf', type=str,
                        help='unbalanced train dataset path')
    parser.add_argument('--eval_hdf5', default='/data/dean/whl/audioset_Kong/mp3/eval_segments_mp3.hdf', type=str,
                        help='eval dataset path')
    parser.add_argument('--norm_file', default='./audioset/mean_std_128.npy', type=str,
                        help='norm file path')
    parser.add_argument('--sample_rate', default=32000, type=int)
    parser.add_argument('--clip_length', default=10, type=int)
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('--in_mem', default=False, type=bool)
    parser.add_argument('--extra_augment', default=True, type=bool)
    parser.add_argument('--roll', default=True, type=bool)
    parser.add_argument('--wavmix', default=True, type=bool)
    parser.add_argument('--specmix', default=True, type=bool)
    parser.add_argument('--mixup_alpha', default=0.3, type=float)
    parser.add_argument('--only_balanced', action='store_true',
                        help='Use balanced audioset for pretrain (debug)')
    parser.set_defaults(only_balanced=False)

    parser.add_argument('--nb_classes', default=527, type=int,
                        help='number of the classification types')
    parser.add_argument('--u_patchout', default=200, type=int,
                        help='number of masked patches')
    parser.add_argument('--epoch_len', default=100000, type=int)

    parser.add_argument('--output_dir', default='./output_dir_finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_finetune',
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

def read_class_indices(csv_file):
    output = []
    with open(csv_file) as f:
        fi = csv.reader(f)
        for row in fi:
            output.append(row[-1])
    return output[1:]


def testmAP(args):
    start_time = time.time()
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
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

    num_tasks = misc.get_world_size() # set datasets and dataloaders
    global_rank = misc.get_rank()
    dataset_val = get_test_set(
        eval_hdf5=args.eval_hdf5, 
        sample_rate=args.sample_rate, 
        clip_length=args.clip_length)
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    print("Sampler_val = %s" % str(sampler_val))

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    assert args.model_type == 'vit' or args.model_type == 'swin', "Only support vit and swin models now."
    if args.model_type == 'vit':
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            norm_file=args.norm_file
        )
    elif args.model_type == 'swin':
        model = models_swinTrans.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            norm_file=args.norm_file
        )

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = BinaryCrossEntropy(smoothing=args.smoothing, reduction='mean')

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    test_stats = evaluate(data_loader_val, model, device)
    print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.3f}%")
    print(f"mAUC of the network on the {len(dataset_val)} test images: {test_stats['mAUC']:.3f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time {}'.format(total_time_str))

def testsingleFile(args):
    start_time = time.time()
    indices = read_class_indices(args.csv_file)
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    assert args.model_type == 'vit' or args.model_type == 'swin', "Only support vit and swin models now."
    if args.model_type == 'vit':
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            norm_file=args.norm_file
        )
    elif args.model_type == 'swin':
        model = models_swinTrans.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            norm_file=args.norm_file
        )

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    print("Resume checkpoint %s" % args.resume)

    (audio, _) = librosa.core.load(args.test_file, sr=32000, mono=True)
    audio_length = round(32000 * 10)
    if len(audio) <= audio_length:
        audio = np.concatenate((audio, np.zeros(audio_length - len(audio))), axis=0)
    else:
        audio = audio[0: audio_length]
    audio = audio.reshape(1, 1, -1)
    audio = torch.Tensor(audio).to(device)

    with torch.no_grad():
        model_without_ddp.eval()
        with torch.cuda.amp.autocast():
            output, _, embs = model_without_ddp(audio, None, specmix=False, inference=True)
            output = torch.sigmoid(output).cpu().detach()
            top_indexes = np.argmax(output, axis=-1)
            sorted_indexes = np.argsort(output)
            embs = embs.cpu().numpy()
    print_out = [indices[sorted_indexes[0][-k-1]] for k in range(args.topk)]
    print('Top {} sound events: {}'.format(args.topk, print_out))
    print('Top 1 sound event: {}'.format(indices[top_indexes[0]]))
    print('embedding: {}'.format(embs.shape))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.test_mode == 'mAP':
        testmAP(args)
    elif args.test_mode == 'single':
        testsingleFile(args)
