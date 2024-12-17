import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.models_swin import SwinTransformer
from models.models_mae import AugmentMelSTFT

class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, n_mels=64, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10, fmax_aug_range=2000,
                 norm_file='mean_std.npy', encoder_stride=32, norm_pix_loss=False, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

        # --------------------------------------------------------------------------
        # Mel Spectrogram
        self.mel = AugmentMelSTFT(
            n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hopsize, n_fft=n_fft, freqm=freqm, timem=timem,
            htk=htk, fmin=fmin, fmax=fmax, norm=norm, fmin_aug_range=fmin_aug_range, fmax_aug_range=fmax_aug_range)
        mean_std_file = np.load(norm_file, allow_pickle=True).item()
        self.frame_mean = torch.Tensor(mean_std_file['frame_mean']).cuda()
        self.frame_std = torch.Tensor(mean_std_file['frame_std']).cuda()
        # --------------------------------------------------------------------------
        self.encoder_stride = encoder_stride
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_features,
                out_channels=self.encoder_stride ** 2, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        self.norm_pix_loss = norm_pix_loss
        
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0.
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward_features(self, x, mask_ratio):
        x = self.patch_embed(x)
        mask = self.random_masking(x, mask_ratio)
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        x = x.reshape(B, C, self.img_size[0] // self.encoder_stride, self.img_size[1] // self.encoder_stride)
        return x, mask
    
    def inference(self, x):
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
        return x
    
    def forward(self, x, mask_ratio):
        x = x.type(torch.HalfTensor).cuda()
        x = self.mel_forward(x)
        x = self.reshape_wav2img(x)
        z, mask = self.forward_features(x, mask_ratio)
        x_rec = self.decoder(z)
        # mask = (1. - mask).reshape(x.shape[0], self.patches_resolution[0], self.patches_resolution[1])
        mask = mask.reshape(x.shape[0], self.patches_resolution[0], self.patches_resolution[1])
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        if self.norm_pix_loss:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.e-6)**.5
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss, x_rec, mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}
    

def simMIM_base(**kwargs):
    model = SwinTransformerForSimMIM(
        img_size=(64, 1024), patch_size=4, in_chans=1, num_classes=527, 
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32], window_size=8,
        mask_ratio=0.75, **kwargs)
    return model
