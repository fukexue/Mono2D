from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class VITUp_1(nn.Module):
    def __init__(self, embed_dim):
        super(VITUp_1, self).__init__()
        self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                                  Norm2d(embed_dim),
                                  nn.GELU(),
                                  nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2), )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = []

        # xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            features.append(ops[i](x))

        return features[0]


class VITUp_2(nn.Module):
    def __init__(self, embed_dim):
        super(VITUp_2, self).__init__()
        self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                                  Norm2d(embed_dim),
                                  nn.GELU(),
                                  nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))

    def forward(self, x):

        out = self.fpn1(x[0])

        return out



class VITUp(nn.Module):
    def __init__(self, embed_dim, out_channel=64):
        super(VITUp, self).__init__()
        self.fpn1 = nn.Conv2d(embed_dim, out_channel, kernel_size=1)
        self.fpn2 = nn.Conv2d(embed_dim, out_channel, kernel_size=1)
        self.fpn3 = nn.Conv2d(embed_dim, out_channel, kernel_size=1)
        self.fpn4 = nn.Conv2d(embed_dim, out_channel, kernel_size=1)

        self.out_channel = out_channel

    def forward(self, x):

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        features = []
        for i in range(len(ops)):
            features.append(ops[i](x[i]))

        return features[0]

