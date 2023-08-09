#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
from models.master import *
from functools import partial
nonlinearity = partial(F.relu, inplace=True)
class EIANet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, num_filters=64, dropout=False, rate=0.1,
                 bn=False):
        super(EIANet, self).__init__()
        bottleneck_width = 32
        avd = False
        cardinality = 1
        norm_layer = nn.InstanceNorm2d
        dilation = 1
        self.deep_supervision = deep_supervision
        self.vgg_features = torchvision.models.vgg19(pretrained=True).features
        self.vgg_features[0] = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.inc = self.vgg_features[:4]
        self.down0_e = nn.Sequential(self.vgg_features[:4])
        self.atten0 = atten_down(64)
        self.down1 = nn.Sequential(self.vgg_features[4:9])
        self.down1_e = nn.Sequential(self.vgg_features[4:9])
        self.atten1 = atten_down(128)
        self.down2 = nn.Sequential(self.vgg_features[9:18])
        self.down3 = nn.Sequential(self.vgg_features[18:27])
        self.down4 = nn.Sequential(self.vgg_features[27:36])
        self.down1_sc = Bottleneck_e(128, 128, cardinality=cardinality,
                                       bottleneck_width=bottleneck_width,
                                       avd=avd, dilation=dilation, norm_layer=norm_layer)
        self.down2_sc = Bottleneck_e(256, 256, cardinality=cardinality,
                                       bottleneck_width=bottleneck_width,
                                       avd=avd, dilation=dilation, norm_layer=norm_layer)
        self.up4 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up2 = up_edge(128*3, 64)
        self.up1 = up_edge(64*3, 64)
        self.outc = outconv(64, n_classes, dropout, rate)

        self.dual_down2 = dual_down(64, 128)
        self.dual_down3 = dual_down(128, 256)
        self.dual_down4 = dual_down(256, 512)
        self.dual_down5 = dual_down(512, 1024)

        self.dsoutc4 = outconv(1024, n_classes)
        self.dsoutc3 = outconv(512, n_classes)
        self.dsoutc2 = outconv(256, n_classes)
        self.dsoutc1 = outconv(128, n_classes)
        self.dsoutc5 = outconv(512 + 512, n_classes)
    def forward(self, x):
        ### downsample ###
        x1 = self.inc(x)
        x_size = x.size()
        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()  # 注意.cuda()
        x1_e = self.down0_e(canny)
        x2 = self.down1(x1)
        x2 = self.down1_sc(x2, x1_e) + x2
        x2_e=self.down1_e(x1)
        x3 = self.down2(x2)
        x3 = self.down2_sc(x3, x2_e) + x3
        x4 = self.down3(x3)
        x5=self.down4(x4)

        ### upsample ###
        x44 = self.up4(x5, x4)
        x33 = self.up3(x44, x3)
        x22 = self.up2(x33, x2,x2_e)
        x11 = self.up1(x22, x1,x1_e)
        x11_con = x11
        x0 = self.outc(x11_con)
        x22_d = self.dual_down2(x11,x22)
        x33_d = self.dual_down3(x22_d, x33)
        x44_d = self.dual_down4(x33_d, x44)
        x55_d = self.dual_down5(x44_d, x5)
        ### reduce_channel ###
        x22_d = self.dsoutc1(x22_d)
        x33_d = self.dsoutc2(x33_d)
        x44_d = self.dsoutc3(x44_d)
        x55_d = self.dsoutc4(x55_d)
        return x0,x22_d,x33_d,x44_d,x55_d
class Conv3(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(Conv3, self).__init__()
        self.k2 = nn.Sequential(
                    # nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x
        # print("identity", identity.shape)
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        #print("sccov3_out", out.shape)
        temp = self.k3(out)
        #print("temp", temp.shape)
        out = torch.mul(temp, out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out
class Conv3_e(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(Conv3_e, self).__init__()
        self.k2 = nn.Sequential(
                    # nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x,x_e):
        identity = x
        # print("identity", identity.shape)
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x_e), identity.size()[2:]))) # sigmoid(identity + k2)
        #print("sccov3_out", out.shape)
        temp = self.k3(out)
        #print("temp", temp.shape)
        out = torch.mul(temp, out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out
class Bottleneck_e(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None):
        self.planes=planes
        super(Bottleneck_e, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.conv1_c = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_c = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv3_b = Conv3(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)
        self.scconv3_c = Conv3_e(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 3, planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.dilation = dilation
        self.stride = stride

    def forward(self, x,x_e):
        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_a = self.relu(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_b = self.relu(out_b)
        out_c = self.conv1_c(x)
        out_c = self.bn1_c(out_c)
        out_c = self.relu(out_c)
        out_k1 = self.k1(out_a)
        out_k1 = self.relu(out_k1)
        out_b = self.scconv3_b(out_b)
        out_b = self.relu(out_b)
        out_c = self.scconv3_c(out_c,x_e)
        out_c = self.relu(out_c)
        if self.avd:
            out_b = self.avd_layer(out_b)
        out = self.conv3(torch.cat([out_k1, out_b, out_c], dim=1))
        out = self.bn3(out)
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


if __name__ == '__main__':
    ras = EIANet(n_channels=1, n_classes=1).cuda()
    input_tensor = torch.randn(4, 1, 256, 256).cuda()
    x0, x_14, x_25, x_36, x_47, ss = ras(input_tensor)
    print(x0.shape)
