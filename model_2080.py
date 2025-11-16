import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import einops

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class StackResBlock(nn.Module):
    def __init__(self,out_channel,num_res=2):
        super(StackResBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class CrossScaleModulator(nn.Module):
    def __init__(self, in_ch=32, out_ch=16):
        super().__init__()
        self.up=nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch,kernel_size=3, padding=1)
        )
        self.conv=nn.Sequential(
            ResBlock(out_ch, out_ch),
            nn.Conv2d(out_ch, out_ch*2, 1)
        )

        self.highpass_filter=ResBlock(out_ch, out_ch)

        self.final_fuse=nn.Conv2d(out_ch*2, out_ch, 3, padding=1)

    def forward(self, xg, xl):
        xg=self.up(xg)
        alpha, beta = torch.chunk(self.conv(xg), 2, dim=1)
        xl = xl*alpha+beta
        xg = xg + self.highpass_filter(xl)

        out=self.final_fuse(torch.cat([xg, xl], dim=1))

        return out, xg

class ConcatConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv=nn.Conv2d(out_ch*2, out_ch*2, kernel_size=3, padding=1)

    def forward(self, xg, xl):
        xg = self.up(xg)
        out, xg = torch.chunk(self.conv(torch.cat([xg, xl], dim=1)), 2, dim=1)

        return out, xg



class PUTFormer(nn.Module):
    def __init__(self, base_ch=8, gtb_head=8, ltb_head=(8,4,2,1), gtb_num=4, ltb_num=(1, 1, 1, 1)):
        super().__init__()
        self.inconv=BasicConv(1, base_ch, 3, 1)
        self.downs=nn.ModuleList([
            BasicConv(base_ch, base_ch * 2, kernel_size=3, stride=2),
            BasicConv(base_ch * 2, base_ch * 4, kernel_size=3, stride=2),
            BasicConv(base_ch * 4, base_ch * 8, kernel_size=3, stride=2),
            BasicConv(base_ch * 8, base_ch * 16, kernel_size=3, stride=2)
        ])
        self.GTBs=nn.ModuleList([TBlock(base_ch*16, gtb_head) for i in range(gtb_num)])
        self.LTBs=nn.ModuleList([
            nn.ModuleList([FactorizedTBlock(base_ch * 8, head=ltb_head[0], size1=4, size2=8) for i in range(ltb_num[0])]),
            nn.ModuleList([FactorizedTBlock(base_ch * 4, head=ltb_head[1], size1=4, size2=8) for i in range(ltb_num[1])]),
            nn.ModuleList([FactorizedTBlock(base_ch * 2, head=ltb_head[2], size1=4, size2=8) for i in range(ltb_num[2])]),
            nn.ModuleList([FactorizedTBlock(base_ch, head=ltb_head[3], size1=4, size2=8) for i in range(ltb_num[3])]),

            # nn.ModuleList([nn.Identity() for i in range(ltb_num[0])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[1])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[2])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[3])]),
        ])

        self.out_convs=nn.ModuleList([
            nn.Conv2d(base_ch * 8, 1, 1),
            nn.Conv2d(base_ch * 4, 1, 1),
            nn.Conv2d(base_ch * 2, 1, 1),
            nn.Conv2d(base_ch * 1, 1, 1),
        ])

        self.CSMs=nn.ModuleList([
            CrossScaleModulator(base_ch * 16, base_ch * 8),
            CrossScaleModulator(base_ch * 8, base_ch * 4),
            CrossScaleModulator(base_ch * 4, base_ch * 2),
            CrossScaleModulator(base_ch * 2, base_ch * 1),

            # ConcatConv(base_ch * 16, base_ch * 8),
            # ConcatConv(base_ch * 8, base_ch * 4),
            # ConcatConv(base_ch * 4, base_ch * 2),
            # ConcatConv(base_ch * 2, base_ch * 1),
        ])

    def multiscale_embed(self, x):
        x=self.inconv(x)
        xs=[]

        for down in self.downs:
            xs.append(x)
            x=down(x)

        return xs, x

    def GTB_foward(self,x):
        x = x + positional_encoding_2d_as(x)
        for gtb in self.GTBs:
            x=gtb(x)
        return x

    def forward(self, x):
        xls, xg=self.multiscale_embed(x)
        xg=self.GTB_foward(xg)

        xls.reverse()
        outs=[]
        for (xl, ltbs, csm, out_conv) in zip(xls, self.LTBs, self.CSMs, self.out_convs):
            for ltb in ltbs:
                xl=ltb(xl)
            out, xg = csm(xg, xl)
            out=out_conv(out)
            outs.append(out)
            # xg, _ = csm(xg, xl)
            # out=out_conv(xg)
            # outs.append(out)

        return outs


class PUTFormer_v1(nn.Module):
    def __init__(self, base_ch=16, gtb_head=4, ltb_head=(8,4,2,1), gtb_num=4, ltb_num=(1, 1, 1, 1)):
        super().__init__()
        self.inconv=BasicConv(1, base_ch, 3, 1)
        self.downs=nn.ModuleList([
            BasicConv(base_ch, base_ch * 2, kernel_size=3, stride=2),
            BasicConv(base_ch * 2, base_ch * 4, kernel_size=3, stride=2),
            BasicConv(base_ch * 4, base_ch * 8, kernel_size=3, stride=2),
            BasicConv(base_ch * 8, base_ch * 16, kernel_size=3, stride=2)
        ])
        # self.GTBs=nn.ModuleList([TBlock(base_ch*16, gtb_head) for i in range(gtb_num)])
        self.GTBs = nn.ModuleList([LinearTBlock(base_ch * 16, gtb_head) for i in range(gtb_num)])
        self.LTBs=nn.ModuleList([
            # nn.ModuleList([FactorizedLinearTBlock(base_ch * 8, head=ltb_head[0], size1=4, size2=8) for i in range(ltb_num[0])]),
            # nn.ModuleList([FactorizedLinearTBlock(base_ch * 4, head=ltb_head[1], size1=4, size2=8) for i in range(ltb_num[1])]),
            # nn.ModuleList([FactorizedLinearTBlock(base_ch * 2, head=ltb_head[2], size1=4, size2=8) for i in range(ltb_num[2])]),
            # nn.ModuleList([FactorizedLinearTBlock(base_ch, head=ltb_head[3], size1=4, size2=8) for i in range(ltb_num[3])]),

            nn.ModuleList(
                [LinearTBlock(base_ch * 8, head_num=ltb_head[0]) for i in range(ltb_num[0])]),
            nn.ModuleList(
                [LinearTBlock(base_ch * 4, head_num=ltb_head[1]) for i in range(ltb_num[1])]),
            nn.ModuleList(
                [LinearTBlock(base_ch * 2, head_num=ltb_head[2]) for i in range(ltb_num[2])]),
            nn.ModuleList(
                [LinearTBlock(base_ch, head_num=ltb_head[3]) for i in range(ltb_num[3])]),

            # nn.ModuleList(
            #     [SwinLinearBlock(head_num=ltb_head[0], dim=base_ch * 8, ksize=8, sftsize=0 if i % 2 == 0 else 0)
            #      for i in range(ltb_num[0])]),
            # nn.ModuleList(
            #     [SwinLinearBlock(head_num=ltb_head[0], dim=base_ch * 4, ksize=8, sftsize=0 if i % 2 == 0 else 0)
            #      for i in range(ltb_num[1])]),
            # nn.ModuleList(
            #     [SwinLinearBlock(head_num=ltb_head[0], dim=base_ch * 2, ksize=8, sftsize=0 if i % 2 == 0 else 0)
            #      for i in range(ltb_num[2])]),
            # nn.ModuleList(
            #     [SwinLinearBlock(head_num=ltb_head[0], dim=base_ch * 1, ksize=8, sftsize=0 if i % 2 == 0 else 0)
            #      for i in range(ltb_num[3])]),

            # nn.ModuleList(
            #     [SwinBlock(head_num=ltb_head[0], dim=base_ch * 8, ksize=8, sftsize=0 if i % 2 == 0 else 0)
            #      for i in range(ltb_num[0])]),
            # nn.ModuleList(
            #     [SwinBlock(head_num=ltb_head[0], dim=base_ch * 4, ksize=8, sftsize=0 if i % 2 == 0 else 0)
            #      for i in range(ltb_num[1])]),
            # nn.ModuleList(
            #     [SwinBlock(head_num=ltb_head[0], dim=base_ch * 2, ksize=8, sftsize=0 if i % 2 == 0 else 0)
            #      for i in range(ltb_num[2])]),
            # nn.ModuleList(
            #     [SwinBlock(head_num=ltb_head[0], dim=base_ch * 1, ksize=8, sftsize=0 if i % 2 == 0 else 0)
            #      for i in range(ltb_num[3])]),



            # nn.ModuleList(
            #     [FactorizedTBlock(base_ch * 8, head=ltb_head[0], size1=4, size2=8) for i in range(ltb_num[0])]),
            # nn.ModuleList(
            #     [FactorizedTBlock(base_ch * 4, head=ltb_head[1], size1=4, size2=8) for i in range(ltb_num[1])]),
            # nn.ModuleList(
            #     [FactorizedTBlock(base_ch * 2, head=ltb_head[2], size1=4, size2=8) for i in range(ltb_num[2])]),
            # nn.ModuleList(
            #     [FactorizedTBlock(base_ch, head=ltb_head[3], size1=4, size2=8) for i in range(ltb_num[3])]),

            # nn.ModuleList([nn.Identity() for i in range(ltb_num[0])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[1])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[2])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[3])]),
        ])

        self.out_convs=nn.ModuleList([
            nn.Conv2d(base_ch * 8, 1, 1),
            nn.Conv2d(base_ch * 4, 1, 1),
            nn.Conv2d(base_ch * 2, 1, 1),
            nn.Conv2d(base_ch * 1, 1, 1),
        ])

        self.CSMs=nn.ModuleList([
            # CrossScaleModulator(base_ch * 16, base_ch * 8),
            # CrossScaleModulator(base_ch * 8, base_ch * 4),
            # CrossScaleModulator(base_ch * 4, base_ch * 2),
            # CrossScaleModulator(base_ch * 2, base_ch * 1),

            ConcatConv(base_ch * 16, base_ch * 8),
            ConcatConv(base_ch * 8, base_ch * 4),
            ConcatConv(base_ch * 4, base_ch * 2),
            ConcatConv(base_ch * 2, base_ch * 1),
        ])

    def multiscale_embed(self, x):
        x=self.inconv(x)
        xs=[]

        for down in self.downs:
            xs.append(x)
            x=down(x)

        return xs, x

    def GTB_foward(self,x):
        x = x + positional_encoding_2d_as(x)
        for gtb in self.GTBs:
            x=gtb(x)
        return x

    def forward(self, x):
        xls, xg=self.multiscale_embed(x)
        xg=self.GTB_foward(xg)

        xls.reverse()
        outs=[]
        for (xl, ltbs, csm, out_conv) in zip(xls, self.LTBs, self.CSMs, self.out_convs):
            for ltb in ltbs:
                xl=ltb(xl)
            out, xg = csm(xg, xl)
            out=out_conv(out)
            outs.append(out)
            # xg, _ = csm(xg, xl)
            # out=out_conv(xg)
            # outs.append(out)

        return outs


class PUTFormer_classsify(nn.Module):
    def __init__(self, base_ch=16, gtb_head=4, ltb_head=(8,4,2,1), gtb_num=4, ltb_num=(1, 1, 1, 1), c=21):
        super().__init__()
        self.inconv=BasicConv(1, base_ch, 3, 1)
        self.downs=nn.ModuleList([
            BasicConv(base_ch, base_ch * 2, kernel_size=3, stride=2),
            BasicConv(base_ch * 2, base_ch * 4, kernel_size=3, stride=2),
            BasicConv(base_ch * 4, base_ch * 8, kernel_size=3, stride=2),
            BasicConv(base_ch * 8, base_ch * 16, kernel_size=3, stride=2)
        ])
        self.GTBs=nn.ModuleList([TBlock(base_ch*16, gtb_head) for i in range(gtb_num)])
        self.LTBs=nn.ModuleList([
            nn.ModuleList([FactorizedTBlock(base_ch * 8, head=ltb_head[0], size1=4, size2=8) for i in range(ltb_num[0])]),
            nn.ModuleList([FactorizedTBlock(base_ch * 4, head=ltb_head[1], size1=4, size2=8) for i in range(ltb_num[1])]),
            nn.ModuleList([FactorizedTBlock(base_ch * 2, head=ltb_head[2], size1=4, size2=8) for i in range(ltb_num[2])]),
            nn.ModuleList([FactorizedTBlock(base_ch, head=ltb_head[3], size1=4, size2=8) for i in range(ltb_num[3])]),

            # nn.ModuleList([nn.Identity() for i in range(ltb_num[0])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[1])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[2])]),
            # nn.ModuleList([nn.Identity() for i in range(ltb_num[3])]),
        ])

        self.out_conv=nn.Conv2d(base_ch, c, kernel_size=1)

        self.CSMs=nn.ModuleList([
            CrossScaleModulator(base_ch * 16, base_ch * 8),
            CrossScaleModulator(base_ch * 8, base_ch * 4),
            CrossScaleModulator(base_ch * 4, base_ch * 2),
            CrossScaleModulator(base_ch * 2, base_ch * 1),

            # ConcatConv(base_ch * 16, base_ch * 8),
            # ConcatConv(base_ch * 8, base_ch * 4),
            # ConcatConv(base_ch * 4, base_ch * 2),
            # ConcatConv(base_ch * 2, base_ch * 1),
        ])

    def multiscale_embed(self, x):
        x=self.inconv(x)
        xs=[]

        for down in self.downs:
            xs.append(x)
            x=down(x)

        return xs, x

    def GTB_foward(self,x):
        x = x + positional_encoding_2d_as(x)
        for gtb in self.GTBs:
            x=gtb(x)
        return x

    def forward(self, x):
        wrapped = x
        xls, xg=self.multiscale_embed(x)
        xg=self.GTB_foward(xg)

        xls.reverse()
        for (xl, ltbs, csm, out_conv) in zip(xls, self.LTBs, self.CSMs, self.out_convs):
            for ltb in ltbs:
                xl=ltb(xl)
            out, xg = csm(xg, xl)

        out=self.out_conv(out)

        prob = out

        k = torch.argmax(prob, dim=1, keepdim=True)

        unwrapped = wrapped + k * 2 * torch.pi

        return prob, unwrapped


class EESANet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):
        super(EESANet, self).__init__()
        self.eeb=EEB()
        self.cbr1=nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=(3,3), padding=1),
            SRB(base_ch)
        )
        self.ap1=nn.MaxPool2d(kernel_size=2, stride=2)

        self.cbr2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=(3, 3), padding=1),
            SRB(base_ch*2)
        )
        self.ap2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cbr3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=(3, 3), padding=1),
            SRB(base_ch*4)
        )
        self.ap3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cbr4 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=(3, 3), padding=1),
            SRB(base_ch*8)
        )
        self.ap4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deep=nn.Sequential(
            SRB(base_ch*8),
            ASPP(base_ch * 8),
            PSA(base_ch * 8),
        )

        # up
        self.up4=nn.ConvTranspose2d(base_ch*8, base_ch*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fuse_bn_act4=nn.Sequential(
            nn.Conv2d(base_ch * 16, base_ch * 8, kernel_size=3, padding=1),
            SRB(base_ch*8)
        )

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fuse_bn_act3 = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=3, padding=1),
            SRB(base_ch*4)
        )

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fuse_bn_act2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            SRB(base_ch*2)
        )

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fuse_bn_act1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 1, kernel_size=3, padding=1),
            SRB(base_ch)
        )

        self.out_conv=nn.Conv2d(base_ch+1, out_ch, kernel_size=1)

    def forward(self, x):
        eeb=self.eeb(x)
        f1=self.cbr1(x)
        x=self.ap1(f1)

        f2=self.cbr2(x)
        x=self.ap2(f2)

        f3=self.cbr3(x)
        x=self.ap3(f3)

        f4=self.cbr4(x)
        x=self.ap4(f4)

        x=self.deep(x)

        u4=self.up4(x)
        cat4=torch.cat([u4, f4], dim=1)
        x=self.fuse_bn_act4(cat4)

        u3=self.up3(x)
        cat3=torch.cat([u3, f3], dim=1)
        x=self.fuse_bn_act3(cat3)

        u2 = self.up2(x)
        cat2 = torch.cat([u2, f2], dim=1)
        x = self.fuse_bn_act2(cat2)

        u1=self.up1(x)
        cat1=torch.cat([u1, f1], dim=1)
        x=self.fuse_bn_act1(cat1)

        return self.out_conv(torch.cat([x, eeb], dim=1))


class PSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv=nn.Conv2d(dim, 3*dim, 1)

    def forward(self, x):
        q, k, v = torch.chunk(self.qkv(x), 3, dim=1)
        out=attention(q, k, v, 1)+x
        return out

class ASPP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1=nn.Conv2d(dim, dim, 1)
        self.conv2=nn.Conv2d(dim, dim, 3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(dim, dim, 3, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(dim, dim, 3, dilation=3, padding=3)

        self.fuse=nn.Conv2d(dim*4, dim, 1)

    def forward(self, x):
        f1=self.conv1(x)
        f2=self.conv2(x)
        f3=self.conv3(x)
        f4=self.conv4(x)

        f=torch.cat([f1, f2, f3, f4], dim=1)

        out=self.fuse(f)

        return out


class SRB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(dim)
        self.conv1=nn.Conv2d(dim, dim, 3, padding=1)
        self.bn1=nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(dim)

        self.lrelu=nn.LeakyReLU()

    def forward(self, x):
        x=self.lrelu(self.bn0(self.conv0(x)))

        x=x+self.lrelu(self.bn1(self.conv1(x)))
        x = x + self.lrelu(self.bn2(self.conv2(x)))
        x = x + self.lrelu(self.bn3(self.conv3(x)))

        return x

from kornia.filters.laplacian import Laplacian

class EEB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=Laplacian((7,7))
        self.r = nn.Parameter(torch.scalar_tensor(1., dtype=torch.float32))
        self.b = nn.Parameter(torch.scalar_tensor(0., dtype=torch.float32))

    def forward(self, x):
        out=self.conv(x)*self.r+self.b
        return out


import torch
from typing import Tuple, Optional


@torch.jit.script
def positional_encoding_2d(shape: Tuple[int, int, int], temperature: float = 1e4, scale: float = 2*math.pi,
                           dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
  """Returns the two-dimensional positional encoding as shape [d_model, h, w]"""
  d_model, h, w = shape[-3:]
  i = torch.arange(d_model // 4, dtype=dtype, device=device)
  ys = torch.arange(h, dtype=dtype, device=device) / (h - 1) * scale
  xs = torch.arange(w, dtype=dtype, device=device) / (w - 1) * scale
  t = (temperature ** (4. / d_model * i)).view(-1,1,1,1,1).expand(-1,2,-1,-1,-1)
  u = torch.cat((xs.expand(1, h, w), ys.unsqueeze(-1).expand(1, h, w)), -3) / t
  uu=torch.zeros_like(u)
  uu[:, 0] = u[:, 0].sin()
  uu[:, 1] = u[:, 1].cos()
  return uu.reshape(-1, h, w) # with channel format: sin(x0) sin(y0) cos(x0) cos(y0) sin(x1) ...


@torch.jit.script
def positional_encoding_2d_as(x: torch.Tensor, temperature: float = 1e4, scale: float = 2*math.pi):
  d, h, w = x.shape[-3:]
  return positional_encoding_2d((d, h, w), temperature, scale, x.dtype, x.device).expand_as(x)


def attention(q, k, v, scale, mask=None):
    '''

    Args:
        q: *, N1, c
        k: *, N2, c
        v: *, N2, c
        mask: *, N1, N2
        scale:

    Returns:
        out: *, N1, c
    '''
    atten=q@k.transpose(-2,-1)*scale
    if mask is not None:
        atten = atten.masked_fill(mask < 0, -1e9)
    atten=torch.softmax(atten, dim=-1)
    # atten = torch.tanh(atten)

    # att=atten.detach().cpu().numpy()
    # for i in range(4):
    #     for j in range(256):
    #         if j==136:
    #             plt.figure(f'88')
    #             plt.imshow(att[0, i, j].reshape((16, 16)))
    #             plt.savefig(f'out/{i}_(8,8).png')
                # plt.show()


    out=atten @ v

    # return out, att
    return out


def linear_attention(q, k, v, scale, mask=None):
    '''

    Args:
        q: *, N1, c
        k: *, N2, c
        v: *, N2, c
        mask: *, N1, N2
        scale:

    Returns:
        out: *, N1, c
    '''
    out=q@((k.transpose(-2,-1)*scale)@v)
    return out

class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".
    Args:
        x: (B, D, H, W, C)
    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x



class TBlock(nn.Module):
    '''
    Multi-head attention + Feed-forward network
    '''
    def __init__(self, fea_dim=128, head_num=4, drop=0.0):
        super(TBlock, self).__init__()
        self.fea_dim=fea_dim
        self.head_num=head_num
        self.q=nn.Linear(fea_dim, fea_dim)
        self.kv=nn.Linear(fea_dim, fea_dim*2)
        self.proj = nn.Linear(fea_dim, fea_dim)
        self.norm=nn.LayerNorm(fea_dim)
        self.head_dim = fea_dim // head_num
        self.scale=self.head_dim**-0.5
        self.norm_ffn=nn.LayerNorm(fea_dim)
        self.ffn=Mlp_GEGLU(fea_dim, drop=drop)

    def forward(self, x):
        '''

        :param x: B, C, H, W
        :return: b, c, h, w
        '''
        b, c, h, w= x.shape
        x=einops.rearrange(x, 'b c h w -> b (h w) c')  # b, hw, c
        q=self.q(x)
        k, v=torch.chunk(self.kv(x), 2, -1)

        q = einops.rearrange(q, 'b n (head c) -> b head n c', head=self.head_num)
        k = einops.rearrange(k, 'b n (head c) -> b head n c', head=self.head_num)
        v = einops.rearrange(v, 'b n (head c) -> b head n c', head=self.head_num)

        # att_out, att=attention(q, k, v, scale=self.scale)
        att_out = attention(q, k, v, scale=self.scale)
        x=self.norm(self.proj(einops.rearrange(att_out, 'b head n c -> b n (head c)')))+x
        # x = self.proj(einops.rearrange(att_out, 'b head n c -> b n (head c)')) + x
        # x=self.ffn(self.norm_ffn(x))+x
        x = self.norm_ffn(self.ffn(x))+x
        x=einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # return x, att
        return x

class LinearTBlock(nn.Module):
    '''
    Multi-head attention + Feed-forward network
    '''
    def __init__(self, fea_dim=128, head_num=4, drop=0.0):
        super(LinearTBlock, self).__init__()
        self.fea_dim=fea_dim
        self.head_num=head_num
        self.q=nn.Linear(fea_dim, fea_dim)
        self.kv=nn.Linear(fea_dim, fea_dim*2)
        self.proj = nn.Linear(fea_dim, fea_dim)
        self.norm=nn.LayerNorm(fea_dim)
        self.head_dim = fea_dim // head_num
        self.scale=self.head_dim**-0.5
        self.norm_ffn=nn.LayerNorm(fea_dim)
        self.ffn=Mlp_GEGLU(fea_dim, drop=drop)

    def forward(self, x):
        '''

        :param x: B, C, H, W
        :return: b, c, h, w
        '''
        b, c, h, w= x.shape
        x=einops.rearrange(x, 'b c h w -> b (h w) c')  # b, hw, c
        q=self.q(x)
        k, v=torch.chunk(self.kv(x), 2, -1)

        q = einops.rearrange(q, 'b n (head c) -> b head n c', head=self.head_num)
        k = einops.rearrange(k, 'b n (head c) -> b head n c', head=self.head_num)
        v = einops.rearrange(v, 'b n (head c) -> b head n c', head=self.head_num)

        # att_out, att=attention(q, k, v, scale=self.scale)
        att_out = linear_attention(q, k, v, scale=self.scale)
        x=self.norm(self.proj(einops.rearrange(att_out, 'b head n c -> b n (head c)')))+x
        # x = self.proj(einops.rearrange(att_out, 'b head n c -> b n (head c)')) + x
        # x=self.ffn(self.norm_ffn(x))+x
        x = self.norm_ffn(self.ffn(x))+x
        x=einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # return x, att
        return x

class MultiHeadSpatialWindowAttention(nn.Module):
    def __init__(self, head_num=4, dim=128, ksize=8, sftsize=4, norm_layer=None, temperature=False, weights=True):
        super(MultiHeadSpatialWindowAttention, self).__init__()
        self.ksize=ksize
        self.sftsize=sftsize
        self.norm_layer=norm_layer
        if norm_layer:
            self.norm=nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim) if weights else nn.Identity()
        self.k = nn.Linear(dim, dim) if weights else nn.Identity()
        self.v = nn.Linear(dim, dim) if weights else nn.Identity()

        self.proj = nn.Linear(dim, dim) if weights else nn.Identity()

        self.head_num = head_num

        self.head_dim = dim // head_num
        # self.scale=self.head_dim**-0.5

        if temperature:
            self.temperature = nn.Parameter(torch.ones(head_num, 1, 1))
        else:
            self.temperature = self.head_dim**-0.5

    def forward(self, hq, hk, hv):
        '''
        hq: (B, L, C, Hq, Wq)
        hk: (B, L, C, H, W)
        hv: (B, L, C, H, W)
        '''
        B, L, C, H, W = hq.shape
        if self.sftsize>0:
            hq=torch.roll(hq, shifts=(-self.sftsize, -self.sftsize), dims=(-2,-1))
            hk = torch.roll(hk, shifts=(-self.sftsize, -self.sftsize), dims=(-2, -1))
            hv = torch.roll(hv, shifts=(-self.sftsize, -self.sftsize), dims=(-2, -1))

        hq = hq.flatten(-2, -1).transpose(-2,-1)  # B, L, HW, C
        hk = hk.flatten(-2, -1).transpose(-2,-1)  # B, L, HW, C
        hv = hv.flatten(-2, -1).transpose(-2,-1)  # B, L, HW, C

        if self.norm_layer:
            hq=self.norm(hq)
            hk=self.norm(hk)
            hv=self.norm(hv)
        hq = hq.unflatten(2, (H,W)).reshape(B, L, H//self.ksize, self.ksize, W//self.ksize, self.ksize, C).transpose(3,4).flatten(4,5)  # B, L, H//k, W//k, k*k, C
        hk = hk.unflatten(2, (H,W)).reshape(B, L, H//self.ksize, self.ksize, W//self.ksize, self.ksize, C).transpose(3,4).flatten(4,5)
        hv = hv.unflatten(2, (H,W)).reshape(B, L, H//self.ksize, self.ksize, W//self.ksize, self.ksize, C).transpose(3,4).flatten(4,5)
        q = self.q(hq).reshape(B, L, H//self.ksize, W//self.ksize, self.ksize*self.ksize, self.head_num, C // self.head_num).transpose(4,5)  # B, L, H//k, W//k, h, k*k, C/h
        k = self.k(hk).reshape(B, L, H//self.ksize, W//self.ksize, self.ksize*self.ksize, self.head_num, C // self.head_num).transpose(4,5)
        v = self.v(hv).reshape(B, L, H//self.ksize, W//self.ksize, self.ksize*self.ksize, self.head_num, C // self.head_num).transpose(4,5)

        # mask = shift_mask(H, W, self.ksize, self.sftsize).unsqueeze(-1)  # H//k, W//k, k*k, 1
        # mask = mask @ mask.transpose(-2, -1)  # H//k, W//k, k*k, k*k
        # mask = mask.unsqueeze(2).unsqueeze(0).unsqueeze(0)  # 1, 1, H//k, W//k, 1, k*k, k*k

        mask = shift_mask(H, W, self.ksize, self.sftsize)  # H//k, W//k, k*k, k*k
        mask = mask.unsqueeze(2).unsqueeze(0).unsqueeze(0)  # 1, 1, H//k, W//k, 1, k*k, k*k

        x = attention(q, k, v, self.temperature, mask=mask).transpose(4, 5).reshape(B, L, H//self.ksize, W//self.ksize, self.ksize*self.ksize, C)# B, L, H//k, W//k, k*k, C

        x = self.proj(x)
        x=x.unflatten(-2, (self.ksize,self.ksize)).transpose(3,4).reshape(B,L,H,W,C).permute(0,1,4,2,3)
        if self.sftsize>0:
            x=torch.roll(x, shifts=(self.sftsize, self.sftsize), dims=(-2,-1))
        return x


class SwinBlock(nn.Module):
    def __init__(self, head_num=4, dim=128, ksize=8, sftsize=4):
        super(SwinBlock, self).__init__()
        self.ksize = ksize
        self.sftsize = sftsize
        self.norm = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)

        self.proj = nn.Linear(dim, dim)

        self.head_num = head_num

        self.head_dim = dim // head_num
        self.scale=self.head_dim**-0.5

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = Mlp_GEGLU(dim)

    def forward(self, x_embed):
        '''
            :param x_embed: B, C, H, W
            :return: b, c, h, w
        '''
        b, c, h, w=x_embed.shape
        if self.sftsize>0:
            x_embed = torch.roll(x_embed, shifts=(-self.sftsize, -self.sftsize), dims=(-2, -1))
        x_embed = x_embed.permute(0, 2, 3, 1)  # b, h, w, c
        nx = self.norm(x_embed)
        q = self.q(nx)
        k, v = torch.chunk(self.kv(nx), 2, -1)

        q = q.reshape(b, h//self.ksize, self.ksize, w//self.ksize, self.ksize,
                    self.head_num, c//self.head_num).permute(0,1,3,5,2,4,6)  # b, h//k, w//k, head, k, k, c//head
        q = q.flatten(4,5)  # b, h//k, w//k, head, k*k, c//head
        k = k.reshape(b, h // self.ksize, self.ksize, w // self.ksize, self.ksize,
                      self.head_num, c // self.head_num).permute(0, 1, 3, 5, 2, 4, 6)
        k = k.flatten(4, 5)
        v = v.reshape(b, h // self.ksize, self.ksize, w // self.ksize, self.ksize,
                      self.head_num, c // self.head_num).permute(0, 1, 3, 5, 2, 4, 6)
        v = v.flatten(4, 5)

        mask = shift_mask(h, w, self.ksize, self.sftsize, device='cuda')  # h//k, w//k, k*k, k*k
        mask = mask.unsqueeze(2).unsqueeze(0)  # 1, H//k, W//k, 1, k*k, k*k

        att=attention(q, k, v, self.scale, mask).unflatten(4, (self.ksize, self.ksize))\
            .permute(0, 1, 4, 2, 5, 3, 6).flatten(-2,-1)  # b, h//k, k, w//k, k, c
        att=att.reshape(b, h, w, c)

        x_embed= self.proj(att) + x_embed
        x_embed= self.ffn(self.norm_ffn(x_embed)) + x_embed

        x_embed=x_embed.permute(0, 3, 1, 2)

        return x_embed


class SwinLinearBlock(nn.Module):
    def __init__(self, head_num=4, dim=128, ksize=8, sftsize=4):
        super(SwinLinearBlock, self).__init__()
        self.ksize = ksize
        self.sftsize = sftsize
        self.norm = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)

        self.proj = nn.Linear(dim, dim)

        self.head_num = head_num

        self.head_dim = dim // head_num
        self.scale=self.head_dim**-0.5

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = Mlp_GEGLU(dim)

    def forward(self, x_embed):
        '''
            :param x_embed: B, C, H, W
            :return: b, c, h, w
        '''
        b, c, h, w=x_embed.shape
        if self.sftsize>0:
            x_embed = torch.roll(x_embed, shifts=(-self.sftsize, -self.sftsize), dims=(-2, -1))
        x_embed = x_embed.permute(0, 2, 3, 1)  # b, h, w, c
        nx = self.norm(x_embed)
        q = self.q(nx)
        k, v = torch.chunk(self.kv(nx), 2, -1)

        q = q.reshape(b, h//self.ksize, self.ksize, w//self.ksize, self.ksize,
                    self.head_num, c//self.head_num).permute(0,1,3,5,2,4,6)  # b, h//k, w//k, head, k, k, c//head
        q = q.flatten(4,5)  # b, h//k, w//k, head, k*k, c//head
        k = k.reshape(b, h // self.ksize, self.ksize, w // self.ksize, self.ksize,
                      self.head_num, c // self.head_num).permute(0, 1, 3, 5, 2, 4, 6)
        k = k.flatten(4, 5)
        v = v.reshape(b, h // self.ksize, self.ksize, w // self.ksize, self.ksize,
                      self.head_num, c // self.head_num).permute(0, 1, 3, 5, 2, 4, 6)
        v = v.flatten(4, 5)

        mask = shift_mask(h, w, self.ksize, self.sftsize, device='cuda')  # h//k, w//k, k*k, k*k
        mask = mask.unsqueeze(2).unsqueeze(0)  # 1, H//k, W//k, 1, k*k, k*k

        att=linear_attention(q, k, v, self.scale, mask).unflatten(4, (self.ksize, self.ksize))\
            .permute(0, 1, 4, 2, 5, 3, 6).flatten(-2,-1)  # b, h//k, k, w//k, k, c
        att=att.reshape(b, h, w, c)

        x_embed= self.proj(att) + x_embed
        x_embed= self.ffn(self.norm_ffn(x_embed)) + x_embed

        x_embed=x_embed.permute(0, 3, 1, 2)

        return x_embed

class FactorizedAttention(nn.Module):
    def __init__(self, size1=4, size2=8, head=4, dim=64):
        super().__init__()

        self.q1 = nn.Linear(dim, dim)
        self.k1v1 = nn.Linear(dim, dim * 2)
        self.proj1 = nn.Linear(dim, dim)

        self.q2 = nn.Linear(dim, dim)
        self.k2v2 = nn.Linear(dim, dim * 2)
        self.proj2 = nn.Linear(dim, dim)

        self.head_num = head

        self.head_dim = dim // head
        self.scale = self.head_dim ** -0.5

        self.size1=size1
        self.size2=size2

    def forward(self,x):
        '''

        :param x: b h w c
        :return: b h w c
        '''
        xprime=einops.rearrange(x, 'b (h s1) (w s2) c-> b h w (s1 s2) c', s1=self.size1, s2=self.size2)
        q1=self.q1(xprime)
        k1, v1 = torch.chunk(self.k1v1(xprime), 2, dim=-1)
        q1 = einops.rearrange(q1, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        k1 = einops.rearrange(k1, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        v1 = einops.rearrange(v1, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        att=attention(q1, k1, v1, scale=self.scale)
        att=einops.rearrange(att, 'b h w head n c -> b h w n (head c)', head=self.head_num)
        x_att1=self.proj1(att)+xprime

        x_att1=einops.rearrange(x_att1, 'b h w (s1 s2) c -> b (h s1) (w s2) c', s1=self.size1, s2=self.size2)

        x_att1=einops.rearrange(x_att1, 'b (h s2) (w s1) c -> b h w (s2 s1) c', s1=self.size1, s2=self.size2)
        q2=self.q2(x_att1)
        k2, v2=torch.chunk(self.k2v2(x_att1), 2, dim=-1)
        q2 = einops.rearrange(q2, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        k2 = einops.rearrange(k2, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        v2 = einops.rearrange(v2, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        att = attention(q2, k2, v2, scale=self.scale)
        att = einops.rearrange(att, 'b h w head n c -> b h w n (head c)', head=self.head_num)
        x_att2 = self.proj2(att) + x_att1

        x_att2=einops.rearrange(x_att2, 'b h w (s2 s1) c -> b (h s2) (w s1) c', s1=self.size1, s2=self.size2)

        out = x_att2

        return out

class FactorizedLinearAttention(nn.Module):
    def __init__(self, size1=4, size2=8, head=4, dim=64):
        super().__init__()

        self.q1 = nn.Linear(dim, dim)
        self.k1v1 = nn.Linear(dim, dim * 2)
        self.proj1 = nn.Linear(dim, dim)

        self.q2 = nn.Linear(dim, dim)
        self.k2v2 = nn.Linear(dim, dim * 2)
        self.proj2 = nn.Linear(dim, dim)

        self.head_num = head

        self.head_dim = dim // head
        self.scale = self.head_dim ** -0.5

        self.size1=size1
        self.size2=size2

    def forward(self,x):
        '''

        :param x: b h w c
        :return: b h w c
        '''
        xprime=einops.rearrange(x, 'b (h s1) (w s2) c-> b h w (s1 s2) c', s1=self.size1, s2=self.size2)
        q1=self.q1(xprime)
        k1, v1 = torch.chunk(self.k1v1(xprime), 2, dim=-1)
        q1 = einops.rearrange(q1, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        k1 = einops.rearrange(k1, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        v1 = einops.rearrange(v1, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        att=linear_attention(q1, k1, v1, scale=self.scale)
        att=einops.rearrange(att, 'b h w head n c -> b h w n (head c)', head=self.head_num)
        x_att1=self.proj1(att)+xprime

        x_att1=einops.rearrange(x_att1, 'b h w (s1 s2) c -> b (h s1) (w s2) c', s1=self.size1, s2=self.size2)

        x_att1=einops.rearrange(x_att1, 'b (h s2) (w s1) c -> b h w (s2 s1) c', s1=self.size1, s2=self.size2)
        q2=self.q2(x_att1)
        k2, v2=torch.chunk(self.k2v2(x_att1), 2, dim=-1)
        q2 = einops.rearrange(q2, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        k2 = einops.rearrange(k2, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        v2 = einops.rearrange(v2, 'b h w n (head c) -> b h w head n c', head=self.head_num)
        att = linear_attention(q2, k2, v2, scale=self.scale)
        att = einops.rearrange(att, 'b h w head n c -> b h w n (head c)', head=self.head_num)
        x_att2 = self.proj2(att) + x_att1

        x_att2=einops.rearrange(x_att2, 'b h w (s2 s1) c -> b (h s2) (w s1) c', s1=self.size1, s2=self.size2)

        out = x_att2

        return out

class FactorizedTBlock(nn.Module):
    def __init__(self, dim=64, head=4, size1=4, size2=8):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attention=FactorizedAttention(size1,size2,head, dim)
        self.norm2=nn.LayerNorm(dim)
        self.ffn=Mlp_GEGLU(dim)

    def forward(self, x):
        '''

        :param x: b c h w
        :return: b c h w
        '''
        x=einops.rearrange(x, 'b c h w -> b h w c')
        x=x+self.norm1(self.attention(x))
        x=x+self.norm2(self.ffn(x))
        x=einops.rearrange(x, 'b h w c -> b c h w')

        return x

class FactorizedLinearTBlock(nn.Module):
    def __init__(self, dim=64, head=4, size1=4, size2=8):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attention=FactorizedLinearAttention(size1,size2,head, dim)
        self.norm2=nn.LayerNorm(dim)
        self.ffn=Mlp_GEGLU(dim)

    def forward(self, x):
        '''

        :param x: b c h w
        :return: b c h w
        '''
        x=einops.rearrange(x, 'b c h w -> b h w c')
        x=x+self.norm1(self.attention(x))
        x=x+self.norm2(self.ffn(x))
        x=einops.rearrange(x, 'b h w c -> b c h w')

        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm=nn.LayerNorm(dim)

    def forward(self, x):
        x=einops.rearrange(x, 'b c h w -> b h w c')
        x=self.norm(x)
        x=einops.rearrange(x, 'b h w c -> b c h w')
        return x

def shift_mask(H, W, ksize, sftsize, device='cpu'):
    '''

    Args:
        H:
        W:
        ksize: window size
        sftsize: shift size
        device:

    Returns:
        H//k, W//k, k*k, k*k
        -1 means ignore
        0 means
    '''
    mask=torch.zeros((H, W)).to(device)
    mask[-sftsize:, :-sftsize]=1
    mask[:-sftsize, -sftsize:]=2
    mask[-sftsize:, -sftsize:]=3

    mask=mask.reshape(H//ksize, ksize, W//ksize, ksize).transpose(1,2).flatten(-2).unsqueeze(-1)

    mask=mask-mask.transpose(-2,-1)

    mask=mask.masked_fill(mask!=0, -1)

    return mask



class DEncBlock(nn.Module):
    def __init__(self, in_ch, g=12):
        super().__init__()
        self.fs=nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(in_ch+g*i, g, kernel_size=3, padding=1),
            nn.BatchNorm2d(g),
            nn.ReLU()
        ) for i in range(4)])

    def forward(self, x):
        for i in range(4):
            x=torch.cat([x, self.fs[i](x)], dim=1)
        return x

class DDecBlock(nn.Module):
    def __init__(self, in_ch, g=12):
        super().__init__()
        self.fs=nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(in_ch+g*i, g, kernel_size=3, padding=1),
            nn.BatchNorm2d(g),
            nn.ReLU()
        ) for i in range(4)])

    def forward(self, x):
        feas=[]
        for i in range(4):
            f=self.fs[i](x)
            feas.append(f)
            x=torch.cat([x, f], dim=1)
        out=torch.cat(feas, dim=1)
        return out

class DenseNet(nn.Module):
    def __init__(self, in_ch=1, c=15):
        super().__init__()
        self.in_conv=nn.Conv2d(in_ch, 48, kernel_size=3, padding=1)
        self.DB1=DEncBlock(48)
        self.mp1=nn.MaxPool2d(kernel_size=2)
        self.DB2=DEncBlock(96)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.DB3 = DEncBlock(144)
        self.mp3 = nn.MaxPool2d(kernel_size=2)
        self.DB4 = DEncBlock(192)
        self.mp4 = nn.MaxPool2d(kernel_size=2)
        self.DB5 = DEncBlock(240)
        self.mp5 = nn.MaxPool2d(kernel_size=2)
        self.bottle_neck=DDecBlock(288)
        self.up1 = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.DB6 = DDecBlock(336)
        self.up2 = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.DB7 = DDecBlock(288)
        self.up3 = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.DB8 = DDecBlock(240)
        self.up4 = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.DB9 = DDecBlock(192)
        self.up5 = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.DB10 = DEncBlock(144)

        self.out_conv=nn.Conv2d(192, c, kernel_size=1)
        self.c=c

    def forward(self, x):
        wrapped=x
        x=self.in_conv(x)

        f1=self.DB1(x)
        x=self.mp1(f1)

        f2 = self.DB2(x)
        x = self.mp2(f2)

        f3 = self.DB3(x)
        x = self.mp3(f3)

        f4 = self.DB4(x)
        x = self.mp4(f4)

        f5 = self.DB5(x)
        x = self.mp5(f5)

        x=self.bottle_neck(x)

        up1=self.up1(x)
        x=self.DB6(torch.cat([f5, up1], dim=1))

        up2=self.up2(x)
        x = self.DB7(torch.cat([f4, up2], dim=1))

        up3 = self.up3(x)
        x = self.DB8(torch.cat([f3, up3], dim=1))

        up4 = self.up4(x)
        x = self.DB9(torch.cat([f2, up4], dim=1))

        up5 = self.up5(x)
        x = self.DB10(torch.cat([f1, up5], dim=1))

        out=self.out_conv(x)

        prob=out

        k=torch.argmax(prob, dim=1, keepdim=True)-self.c//2

        unwrapped=wrapped+k*2*torch.pi

        return prob, unwrapped


if __name__ == '__main__':
    from tqdm import tqdm
    # net=PUNet_v8(base_ch=8, blk_num=4).cuda()
    # net = DenseNet(c=21).cuda()
    # net = PUTFormer_v1(base_ch=16, gtb_head=8, ltb_head=(8,4,2,1), gtb_num=4, ltb_num=(4,2,2,1)).cuda()
    net = PUTFormer(base_ch=8, gtb_head=8, ltb_head=(8, 4, 2, 1), gtb_num=4, ltb_num=(1,1,1,1)).cuda()
    # net=EESANet(base_ch=42).cuda()
    total_samples=1000
    batch_size=10
    # batch_size = 20
    input=torch.rand(batch_size,1,256,256).cuda()
    from thop import profile
    flops, params = profile(net, [input])
    print("Number of parameter: %.2fM" % (params / 1e6))
    print(f"FLOPs:{flops / 1e9:.2f}G")
    total_time=0
    iters=total_samples//batch_size
    for i in tqdm(range(iters)):
        input = torch.rand(batch_size, 1, 256, 256).cuda()
        torch.cuda.synchronize()
        start=time.time()
        with torch.no_grad():
            out = net(input)
        torch.cuda.synchronize()
        # if i>burn:
        #     total_time+=time.time()-start
        total_time += time.time() - start
    print(total_time/total_samples)