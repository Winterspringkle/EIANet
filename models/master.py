import numpy as np
import torch
from torchvision import models
import torch.nn as nn
# from resnet import resnet34
# import resnet
from torch.nn import functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, reduction=16):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        # x = self.se(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x

class up_edge(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_edge, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.sigmoid = nn.Sigmoid()
        self.change_ch = nn.Conv2d(int(in_ch), int(in_ch/2), kernel_size=1)
    def forward(self, x1, x2,edge):
        #x1:Decoder x2:Encoder,a_map edge
        # print("x1", x1.size())
        # print("x2", x2.size())
        # print("a_map", a_map.size())

        # print("a_map1", a_map.size())
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([edge,x2, x1], dim=1)
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.sigmoid = nn.Sigmoid()
        self.change_ch = nn.Conv2d(int(in_ch), int(in_ch/2), kernel_size=1)
    def forward(self, x1, x2):
        # print("x1", x1.size())
        # print("x2", x2.size())
        # print("a_map", a_map.size())
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        if x2.shape[1]!=x1.shape[1]:
            x1=self.change_ch(x1)
        # print("x2", x2.shape)
        # print("x1", x1.shape)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, rate=0.1):
        super(outconv, self).__init__()
        self.dropout = dropout
        if dropout:
            print('dropout', rate)
            self.dp = nn.Dropout2d(rate)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        if self.dropout:
            x = self.dp(x)
        x = self.conv(x)
        return x

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class dual_down(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(dual_down, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3,2,autopad(3, 1),groups=1),nn.ReLU(),nn.Dropout2d())
        self.conv2 = nn.Sequential(nn.Conv2d(2*in_ch, out_ch, 1), nn.ReLU(), nn.Dropout2d())
    def forward(self, x1, x2):
        x1=self.conv1(x1)
        # print("x1",x1.shape,"x2",x2.shape)
        x=torch.cat([x1,x2],dim=1)
        x=self.conv2(x)
        return x

class atten_down(nn.Module):
    def __init__(self, in_ch):
        super(atten_down, self).__init__()
        self.edge_atten = nn.Sequential(nn.Conv2d(in_ch,in_ch,kernel_size=3, padding=1),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, mask, edge):
        e_atten=self.edge_atten(edge)
        mask=self.act(self.bn(self.edge_atten(mask)))
        mask=mask*e_atten

        return mask
