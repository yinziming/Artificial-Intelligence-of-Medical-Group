import torch
import torch.nn as nn
from torchsummary import summary

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU6()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvDwLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.activation1 = nn.ReLU6()
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation2 = nn.ReLU6()
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.activation2(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()
        self.conv1 = ConvLayer(in_channels, 32, 3, stride=2, padding=1)
        self.convdw1 = ConvDwLayer(32, 32, 3, stride=1, padding=1)
        self.conv2 = ConvLayer(32, 64, 1, stride=1, padding=0)
        self.convdw2 = ConvDwLayer(64, 64, 3, stride=2, padding=1)
        self.conv3 = ConvLayer(64, 128, 1, stride=1, padding=0)
        self.convdw3 = ConvDwLayer(128, 128, 3, stride=1, padding=1)
        self.conv4 = ConvLayer(128, 128, 1, stride=1, padding=0)
        self.convdw4 = ConvDwLayer(128, 128, 3, stride=2, padding=1)
        self.conv5 = ConvLayer(128, 256, 1, stride=1, padding=0)
        self.convdw5 = ConvDwLayer(256, 256, 3, stride=1, padding=1)
        self.conv6 = ConvLayer(256, 256, 1, stride=1, padding=0)
        self.convdw6 = ConvDwLayer(256, 256, 3, stride=2, padding=1)
        self.conv7 = ConvLayer(256, 512, 1, stride=1, padding=0)
        
        # 5xconvdw 5xconv
        self.convdw7 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv8 = ConvLayer(512, 512, 1, stride=1, padding=0)
        self.convdw8 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv9 = ConvLayer(512, 512, 1, stride=1, padding=0)
        self.convdw9 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv10 = ConvLayer(512, 512, 1, stride=1, padding=0)
        self.convdw10 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv11 = ConvLayer(512, 512, 1, stride=1, padding=0)
        self.convdw11 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv12 = ConvLayer(512, 512, 1, stride=1, padding=0)
        
        self.convdw12 = ConvDwLayer(512, 512, 3, stride=2, padding=1)
        self.conv13 = ConvLayer(512, 1024, 1, stride=1, padding=0)
        self.convdw13 = ConvDwLayer(1024, 1024, 3, stride=1, padding=1)
        self.conv14 = ConvLayer(1024, 1024, 1, stride=1, padding=0)
        
        # classification layers
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, n_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.convdw1(x)
        x = self.conv2(x)
        x = self.convdw2(x)
        x = self.conv3(x)
        x = self.convdw3(x)
        x = self.conv4(x)
        x = self.convdw4(x)
        x = self.conv5(x)
        x = self.convdw5(x)
        x = self.conv6(x)
        x = self.convdw6(x)
        x = self.conv7(x)
        x = self.convdw7(x)
        x = self.conv8(x)
        x = self.convdw8(x)
        x = self.conv9(x)
        x = self.convdw9(x)
        x = self.conv10(x)
        x = self.convdw10(x)
        x = self.conv11(x)
        x = self.convdw11(x)
        x = self.conv12(x)
        x = self.convdw12(x)
        x = self.conv13(x)
        x = self.convdw13(x)
        x = self.conv14(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def mobile_net(in_channels=1, out_channels=2):
    net = MobileNet(in_channels, out_channels)
    return net

if __name__ == "__main__":
    net = mobile_net(1, 2)
    print(net)
    summary(net.cuda(), (1, 100, 272, 256), batch_size=1)