import torch
import torch.nn as nn
from functools import partial
from res_se_unet_3D_V1 import SELayer_3D

class Conv_Norm_Activation_3D(nn.Module):
    def __init__(self, feat_in, feat_out, kernel_size=3, stride=2, padding=1, bias=False, dilation=1,
                 Norm=nn.InstanceNorm3d, Activation=nn.LeakyReLU, dropout=0.0):
        super(Conv_Norm_Activation_3D, self).__init__()
        self.CNA = nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
                      dilation=dilation),
            Norm(feat_out),
            nn.Dropout(p=dropout, inplace=True),
            Activation())

    def forward(self, x):
        x = self.CNA(x)
        return x

class UNet_3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=(32, 64, 128, 256, 512), norm='batch',
                 act='relu'):#320
        super(UNet_3D, self).__init__()
        if norm == 'instance':
            norm = partial(nn.InstanceNorm3d, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        elif norm == 'batch':
            norm = nn.BatchNorm3d

        if act == 'leakyrelu':
            act = partial(nn.LeakyReLU, inplace=True, negative_slope=0.01)
        elif act == 'relu':
            act = partial(nn.ReLU, inplace=True)

        self.up_3_attention = SELayer_3D(features[3])
        self.up_2_attention = SELayer_3D(features[2])
        self.up_1_attention = SELayer_3D(features[1])
        self.up_0_attention = SELayer_3D(features[0])

        self.encoder_0 = nn.Sequential(
            Conv_Norm_Activation_3D(in_channels, features[0], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act),
            Conv_Norm_Activation_3D(features[0], features[0], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )
        self.encoder_1 = nn.Sequential(
            Conv_Norm_Activation_3D(features[0], features[1], kernel_size=3, stride=2, padding=1, Norm=norm,
                                    Activation=act),
            Conv_Norm_Activation_3D(features[1], features[1], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )
        self.encoder_2 = nn.Sequential(
            Conv_Norm_Activation_3D(features[1], features[2], kernel_size=3, stride=2, padding=1, Norm=norm,
                                    Activation=act),
            Conv_Norm_Activation_3D(features[2], features[2], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )
        self.encoder_3 = nn.Sequential(
            Conv_Norm_Activation_3D(features[2], features[3], kernel_size=3, stride=2, padding=1, Norm=norm,
                                    Activation=act),
            Conv_Norm_Activation_3D(features[3], features[3], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )
        self.encoder_4 = nn.Sequential(
            Conv_Norm_Activation_3D(features[3], features[4], kernel_size=3, stride=2, padding=1, Norm=norm,
                                    Activation=act),
            Conv_Norm_Activation_3D(features[4], features[4], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )

        self.up_3 = nn.ConvTranspose3d(features[4], features[3], 2, 2, bias=False)
        self.decoder_3 = nn.Sequential(
            Conv_Norm_Activation_3D(features[3] + features[3], features[3], kernel_size=3, stride=1,
                                    padding=1, Norm=norm, Activation=act),
            Conv_Norm_Activation_3D(features[3], features[3], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )
        self.up_2 = nn.ConvTranspose3d(features[3], features[2], 2, 2, bias=False)
        self.decoder_2 = nn.Sequential(
            Conv_Norm_Activation_3D(features[2] + features[2], features[2], kernel_size=3, stride=1,
                                    padding=1, Norm=norm, Activation=act),
            Conv_Norm_Activation_3D(features[2], features[2], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )
        self.up_1 = nn.ConvTranspose3d(features[2], features[1], 2, 2, bias=False)
        self.decoder_1 = nn.Sequential(
            Conv_Norm_Activation_3D(features[1] + features[1], features[1], kernel_size=3, stride=1,
                                    padding=1, Norm=norm, Activation=act),
            Conv_Norm_Activation_3D(features[1], features[1], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )
        self.up_0 = nn.ConvTranspose3d(features[1], features[0], 2, 2, bias=False)
        self.decoder_0 = nn.Sequential(
            Conv_Norm_Activation_3D(features[0] + features[0], features[0], kernel_size=3, stride=1,
                                    padding=1, Norm=norm, Activation=act),
            Conv_Norm_Activation_3D(features[0], features[0], kernel_size=3, stride=1, padding=1, Norm=norm,
                                    Activation=act)
        )

        self.out = nn.Conv3d(features[0], out_channels, 1, 1, bias=False)

    def forward(self, x):
        x_map0 = self.encoder_0(x)
        x_map1 = self.encoder_1(x_map0)
        x_map2 = self.encoder_2(x_map1)
        x_map3 = self.encoder_3(x_map2)
        x = self.encoder_4(x_map3)

        x = self.up_3(x)
        #x = self.up_3_attention(x)
        x = torch.cat((x, x_map3), dim=1)
        x = self.decoder_3(x)

        x = self.up_2(x)
        #x = self.up_2_attention(x)
        x = torch.cat((x, x_map2), dim=1)
        x = self.decoder_2(x)

        x = self.up_1(x)
        #x = self.up_1_attention(x)
        x = torch.cat((x, x_map1), dim=1)
        x = self.decoder_1(x)

        x = self.up_0(x)
        #x = self.up_0_attention(x)
        x = torch.cat((x, x_map0), dim=1)
        x = self.decoder_0(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    net = UNet_3D(in_channels=1, out_channels=2, features=(8, 16, 32, 32, 32), norm='instance',
                 act='leakyrelu')
    x = torch.rand((1, 1, 128, 128, 160))
    y = net(x)
    print(y.size())