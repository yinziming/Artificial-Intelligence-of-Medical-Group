import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchsummary import summary

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout) -> None:
        super(Down, self).__init__()
        self.resblock = Residual_Down(in_channels, dropout)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        conv = self.resblock(x)
        output = self.conv(conv)
        return conv, output

class Bridge(nn.Module):
    def __init__(self, channels, dropout) -> None:
        super(Bridge, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y += x
        return y

class Up(nn.Module):
    def __init__(self, size, in_channels, out_channels, dropout) -> None:
        super(Up, self).__init__()
        if size == None:
            self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up_sample = nn.Upsample(size=size, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.resblock = Residual_Up(out_channels, dropout)

    def forward(self, input_down, x):
        x = self.up_sample(x)
        x = self.conv(x)
        x = self.resblock(input_down, x)
        return x

class Residual_Down(nn.Module):
    def __init__(self, channels, dropout) -> None:
        super(Residual_Down, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.dropout = nn.Dropout3d(dropout)
        self.channels = channels
    
    def forward(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y += x
        return y

class Residual_Up(nn.Module):
    def __init__(self, channels, dropout) -> None:
        super(Residual_Up, self).__init__()
        self.conv1 = nn.Conv3d(channels*2, channels, kernel_size=1)
        self.conv2 = nn.Conv3d(channels*2, channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(channels*2)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, input_down, x):
        x = torch.cat((x, input_down), dim=1)
        y = self.bn1(x)
        x = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.dropout(y)
        y = self.conv3(y)
        y += x
        return y

class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels, is_train=True, size_list=None) -> None:
        super(ResUnet, self).__init__()
        channels = 64
        dropout = 0.2
        self.is_train = is_train
        self.input = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1, stride=1)
        self.down1 = Down(channels, channels*2, dropout)
        self.down2 = Down(channels*2, channels*4, dropout)
        self.down3 = Down(channels*4, channels*8, dropout)
        self.bridge = Bridge(channels*8, dropout)
        if size_list == None:
            self.up3 = Up(size_list, channels*8, channels*4, dropout)
            self.up2 = Up(size_list, channels*4, channels*2, dropout)
            self.up1 = Up(size_list, channels*2, channels, dropout)
        else:
            self.up3 = Up(size_list[0], channels*8, channels*4, dropout)
            self.up2 = Up(size_list[1], channels*4, channels*2, dropout)
            self.up1 = Up(size_list[2], channels*2, channels, dropout)

        self.out = nn.Sequential(nn.BatchNorm3d(channels), nn.PReLU(), nn.Conv3d(channels, out_channels, kernel_size=1), nn.Dropout3d(dropout), nn.Softmax(dim=1))
        # self.out = nn.Sequential(nn.BatchNorm3d(channels), nn.PReLU(), nn.Conv3d(channels, out_channels, kernel_size=1), nn.Dropout3d(dropout))

    def forward(self, x):
        if self.is_train:
            x = x + torch.normal(0, 0.01, x.shape, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        x = self.input(x)
        input_down1, x = checkpoint(self.down1,x)
        input_down2, x = checkpoint(self.down2,x)
        input_down3, x = checkpoint(self.down3,x)
        x = checkpoint(self.bridge,x)
        x = checkpoint(self.up3,input_down3, x)
        x = checkpoint(self.up2,input_down2, x)
        x = checkpoint(self.up1,input_down1, x)
        x = checkpoint(self.out,x)
        return x


if __name__ == "__main__":
    in_channels, out_channels = 4, 4
    size_list = ((38, 60, 60), (77, 120, 120), (155, 240, 240))
    net = ResUnet(in_channels, out_channels, size_list=size_list)
    summary(net.cuda(), (4, 155, 240, 240))