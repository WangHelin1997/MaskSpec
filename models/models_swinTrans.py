import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import models.models_swin
from models.models_mae import AugmentMelSTFT

class SwinTransformer(models.models_swin.SwinTransformer):
    def __init__(self, n_mels=64, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10, fmax_aug_range=2000,
                 norm_file='mean_std.npy', **kwargs):
        super().__init__(**kwargs)
        # --------------------------------------------------------------------------
        # Mel Spectrogram
        self.mel = AugmentMelSTFT(
            n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hopsize, n_fft=n_fft, freqm=freqm, timem=timem,
            htk=htk, fmin=fmin, fmax=fmax, norm=norm, fmin_aug_range=fmin_aug_range, fmax_aug_range=fmax_aug_range)
        mean_std_file = np.load(norm_file, allow_pickle=True).item()
        self.frame_mean = torch.Tensor(mean_std_file['frame_mean']).cuda()
        self.frame_std = torch.Tensor(mean_std_file['frame_std']).cuda()
        # --------------------------------------------------------------------------
        
    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = (x - self.frame_mean[None, :, None]) / self.frame_std[None, :, None]
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x
    
    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        B, C, F, T = x.shape
        target_T = self.img_size[1]
        assert T <= target_T, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (x.shape[2], target_T), mode="bicubic", align_corners=True)
        return x
    
    def mixup(self, size, alpha):
        rn_indices = torch.randperm(size)
        lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
        lam = torch.FloatTensor(lambd)
        return rn_indices, lam
    
    def forward(self, x, y, specmix=True, mixup_alpha=0.3):
        x = x.type(torch.HalfTensor).cuda()
        x = self.mel_forward(x)
        x = self.reshape_wav2img(x)
        B = x.shape[0]
        if self.training and specmix:
            rn_indices, lam = self.mixup(B, mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(B, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(B, 1, 1, 1))
            y = y * lam.reshape(B, 1) + y[rn_indices] * (1. - lam.reshape(B, 1))

        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x, y
    

def swinTrans_base(**kwargs):
    model = SwinTransformer(
        img_size=(64, 1024), patch_size=4, in_chans=1, 
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32], window_size=8,
        **kwargs)
    return model
