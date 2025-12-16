"""
Projected discriminator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm

from typing import Callable

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_feat, out_feat, activation=nn.Identity()):
        super().__init__()
        self.linear_layer = nn.Linear(in_feat, out_feat)
        self.act = activation

    def forward(self, x):
        return self.act(self.linear_layer(x))


class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)
    
        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)

            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        

        return out



class ProjectedDiscriminator(nn.Module):
    def __init__(self, c_dim: int = 512, diffaug: bool = False, embed_dim: int = 768, num_of_disc_feats:int=4, num_of_classes:int=2, device=None):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.embed_dim = embed_dim
        self.num_of_classes = num_of_classes
        self.feat_modulator = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        self.heads = nn.ModuleList()
        for _ in range(num_of_classes):
            heads = []
            for i in range(num_of_disc_feats):
                heads += [str(i), DiscHead(self.embed_dim, c_dim)],
            self.heads.append(nn.ModuleDict(heads))

    def train(self, mode: bool = True):
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

        
    def forward(self, features: torch.Tensor, text_encoding: torch.Tensor, class_label: torch.Tensor = None) -> torch.Tensor:
        logits = []

        if class_label is None:
            class_label = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        
        for i in range(features.size(0)):
            logits_ = []

            label = class_label[i].item()
            for k, head in self.heads[label].items():
                logits_.append(head(features[i:(i+1),int(k),:,:], text_encoding[i:(i+1)]).view(1, -1))
            logits_ = torch.cat(logits_, dim=1)

            logits.append(logits_)

        logits = torch.cat(logits, dim=0)

        return logits

    

