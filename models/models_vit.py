# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.models_mae import AugmentMelSTFT


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10, fmax_aug_range=2000,
                 norm_file='mean_std.npy', u_patchout=0, target_size=(128, 1000), **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # --------------------------------------------------------------------------
        # Mel Spectrogram
        self.mel = AugmentMelSTFT(
            n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hopsize, n_fft=n_fft, freqm=freqm, timem=timem,
            htk=htk, fmin=fmin, fmax=fmax, norm=norm, fmin_aug_range=fmin_aug_range, fmax_aug_range=fmax_aug_range)
        mean_std_file = np.load(norm_file, allow_pickle=True).item()
        self.frame_mean = torch.Tensor(mean_std_file['frame_mean']).cuda()
        self.frame_std = torch.Tensor(mean_std_file['frame_std']).cuda()
        # --------------------------------------------------------------------------
        # Augmentation
        self.u_patchout = u_patchout
        self.target_size = target_size
        # --------------------------------------------------------------------------
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
    
    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = (x - self.frame_mean[None, :, None]) / self.frame_std[None, :, None]
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x
    
    def mixup(self, size, alpha):
        rn_indices = torch.randperm(size)
        lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
        lam = torch.FloatTensor(lambd)
        return rn_indices, lam
    
    def reshape_wav2img(self, x):
        B, C, F, T = x.shape
        target_T = self.patch_embed.img_size[1]
        assert T <= target_T, "the wav size should less than or equal to the input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (x.shape[2], target_T), mode="bicubic", align_corners=True)
        return x

    def forward_features(self, x, y, specmix=True, mixup_alpha=0.3):
        x = x.type(torch.HalfTensor).cuda()
        x = self.mel_forward(x)
        x = self.reshape_wav2img(x)
        # x = x[:, :, :self.patch_embed.img_size[0], :self.patch_embed.img_size[1]]
        B = x.shape[0]
        if self.training and specmix:
            rn_indices, lam = self.mixup(B, mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(B, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(B, 1, 1, 1))
            y = y * lam.reshape(B, 1) + y[rn_indices] * (1. - lam.reshape(B, 1))

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        if self.target_size[1] < self.patch_embed.img_size[1]:
            target_t = self.target_size[1] // self.patch_embed.patch_size[1]
            cls_feature = x[:, 0, :]
            cls_feature = cls_feature[:, None, :]
            audio_feature = x[:, 1:, :]
            audio_feature = audio_feature.reshape(audio_feature.shape[0], self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], -1)
            audio_feature = audio_feature[:, :, :target_t, :]
            audio_feature = audio_feature.reshape(audio_feature.shape[0], -1, audio_feature.shape[-1])
            x = torch.cat((cls_feature, audio_feature), dim=1)
        x = self.pos_drop(x)

        if self.training and self.u_patchout:
            seq_len = x.shape[1] - 1
            cls_tokens = x[:, 0, :]
            cls_tokens = cls_tokens[:, None, :]
            random_indices = torch.randperm(seq_len)[:seq_len - self.u_patchout].sort().values
            x = x[:, 1:, :]
            x = x[:, random_indices, :]
            x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, y
    
    def forward(self, x, y, specmix=True, mixup_alpha=0.3, inference=False):
        x, y = self.forward_features(x, y, specmix, mixup_alpha)
        emb = x
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        if inference:
            return x, y, emb
        else:
            return x, y


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=(128, 1000), patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        img_size=(128, 1000), patch_size=16, in_chans=1, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        img_size=(128, 1000), patch_size=14, in_chans=1, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
