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

class InceptionBlockA(nn.Module):
    def __init__(self, in_channels, fist_block=False) -> None:
        super().__init__()
        if fist_block:
            # branch1
            self.b1_conv = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
            #branch2
            self.b2_conv1 = ConvLayer(in_channels, 48, kernel_size=1, stride=1, padding=0)
            self.b2_conv2 = ConvLayer(48, 64, kernel_size=5, stride=1, padding=2)
            #branch3
            self.b3_conv1 = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
            self.b3_conv2 = ConvLayer(64, 96, kernel_size=3, stride=1, padding=1)
            self.b3_conv3 = ConvLayer(96, 96, kernel_size=3, stride=1, padding=1)
            #branch4
            self.b4_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
            self.b4_conv = ConvLayer(in_channels, 32, kernel_size=1, stride=1, padding=0)
        else:
            # branch1
            self.b1_conv = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
            #branch2
            self.b2_conv1 = ConvLayer(in_channels, 48, kernel_size=1, stride=1, padding=0)
            self.b2_conv2 = ConvLayer(48, 64, kernel_size=5, stride=1, padding=2)
            #branch3
            self.b3_conv1 = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
            self.b3_conv2 = ConvLayer(64, 96, kernel_size=3, stride=1, padding=1)
            self.b3_conv3 = ConvLayer(96, 96, kernel_size=3, stride=1, padding=1)
            #branch4
            self.b4_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
            self.b4_conv = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        b1_y = self.b1_conv(x)
        
        b2_y = self.b2_conv1(x)
        b2_y = self.b2_conv2(b2_y)
        
        b3_y = self.b3_conv1(x)
        b3_y = self.b3_conv2(b3_y)
        b3_y = self.b3_conv3(b3_y)
        
        b4_y = self.b4_pool(x)
        b4_y = self.b4_conv(b4_y)
        
        y = torch.concat((b1_y, b2_y, b3_y, b4_y), dim=1)
        
        return y

class InceptionBlockB(nn.Module):
    def __init__(self, in_channels, block_num=1) -> None:
        super().__init__()
        out_channels = {2:128, 3:160, 4:160, 5:192}
        if block_num == 1:
            # branch1
            self.b1 = nn.Sequential(ConvLayer(in_channels, 384, kernel_size=3, stride=2, padding=0))
            self.b2 = nn.Sequential(ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0),
                                    ConvLayer(64, 96, kernel_size=3, stride=1, padding=1),
                                    ConvLayer(96, 96, kernel_size=3, stride=2, padding=0))
            self.b3 = None
            self.b4 = nn.Sequential(nn.MaxPool3d(kernel_size=3, stride=2, padding=0))
        else:
            self.b1 = nn.Sequential(ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0))
            self.b2 = nn.Sequential(ConvLayer(in_channels, out_channels[block_num], kernel_size=1, stride=1, padding=0),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0)),
                                    ConvLayer(out_channels[block_num], 192, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0)))
            self.b3 = nn.Sequential(ConvLayer(in_channels, out_channels[block_num], kernel_size=1, stride=1, padding=0),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0)),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0)),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0)),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0)),
                                    ConvLayer(out_channels[block_num], 192, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)))
            self.b4 = nn.Sequential(nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                                    ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        temp = x
        for each_layer in self.b1:
            temp = each_layer(temp)
        b1_y = temp
        temp = x
        for each_layer in self.b2:
            temp = each_layer(temp)
        b2_y = temp
        temp = x
        for each_layer in self.b4:
            temp = each_layer(temp)
        b4_y = temp
        temp = x
        if self.b3 is not None:
            for each_layer in self.b1:
                temp = each_layer(temp)
            b3_y = temp
            y = torch.concat((b1_y, b2_y, b3_y, b4_y), dim=1)
        else:
            y = torch.concat((b1_y, b2_y, b4_y), dim=1)
        
        return y

class InceptionBlockC_1(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        # branch1
        self.b1_conv1 = ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0)
        self.b1_conv2 = ConvLayer(192, 320, kernel_size=3, stride=2, padding=0)
        
        #branch2
        self.b2_conv1 = ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0)
        self.b2_conv2 = ConvLayer(192, 192, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3))
        self.b2_conv3 = ConvLayer(192, 192, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0))
        self.b2_conv4 = ConvLayer(192, 192, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0))
        self.b2_conv5 = ConvLayer(192, 192, kernel_size=3, stride=2, padding=0)
        
        #branch3
        self.b3_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        b1_y = self.b1_conv1(x)
        b1_y = self.b1_conv2(b1_y)
        b2_y = self.b2_conv1(x)
        b2_y = self.b2_conv2(b2_y)
        b2_y = self.b2_conv3(b2_y)
        b2_y = self.b2_conv4(b2_y)
        b2_y = self.b2_conv5(b2_y)
        b3_y = self.b3_pool(x)
        
        y = torch.concat((b1_y, b2_y, b3_y), dim=1)
        
        return y
        
class InceptionBlockC_2(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        # branch1
        self.b1_conv = ConvLayer(in_channels, 320, kernel_size=1, stride=1, padding=0)
        
        #branch2
        self.b2_conv1 = ConvLayer(in_channels, 384, kernel_size=1, stride=1, padding=0)
        self.b2_conv2 = ConvLayer(384, 384, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        self.b2_conv3 = ConvLayer(384, 384, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        self.b2_conv4 = ConvLayer(384, 384, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        
        #branch3
        self.b3_conv1 = ConvLayer(in_channels, 448, kernel_size=1, stride=1, padding=0)
        self.b3_conv2 = ConvLayer(448, 384, kernel_size=3, stride=1, padding=1)
        self.b3_conv3 = ConvLayer(384, 384, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        self.b3_conv4 = ConvLayer(384, 384, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        self.b3_conv5 = ConvLayer(384, 384, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        
        #branch4
        self.b4_conv = ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        b1_y = self.b1_conv(x)
        b2_y = self.b2_conv1(x)
        b2_y1 = self.b2_conv2(b2_y)
        b2_y2 = self.b2_conv3(b2_y)
        b2_y = torch.concat((b2_y1, b2_y2), dim=1)
        b3_y = self.b3_conv1(x)
        b3_y = self.b3_conv2(b3_y)
        b3_y1 = self.b3_conv3(b3_y)
        b3_y2 = self.b3_conv4(b3_y)
        b3_y = torch.concat((b3_y1, b3_y2), dim=1)
        b4_y = self.b4_conv(x)
        
        y = torch.concat((b1_y, b2_y, b3_y, b4_y), dim=1)
        
        return y

class Inceptionv3(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.conv4 = ConvLayer(64, 80, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvLayer(80, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        
        self.inception_block_a1 = InceptionBlockA(192, True)
        self.inception_block_a2 = InceptionBlockA(256, False)
        self.inception_block_a3 = InceptionBlockA(288, False)

        self.inception_block_b1 = InceptionBlockB(288, 1)
        self.inception_block_b2 = InceptionBlockB(768, 2)
        self.inception_block_b3 = InceptionBlockB(768, 3)
        self.inception_block_b4 = InceptionBlockB(768, 4)
        self.inception_block_b5 = InceptionBlockB(768, 5)
        
        self.inception_block_c1 = InceptionBlockC_1(768)
        self.inception_block_c2 = InceptionBlockC_2(1280)
        self.inception_block_c3 = InceptionBlockC_2(2048)
        
        self.pool3 = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout3d()
        self.conv6 = ConvLayer(2048, n_classes, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.inception_block_a1(x)
        x = self.inception_block_a2(x)
        x = self.inception_block_a3(x)
        x = self.inception_block_b1(x)
        x = self.inception_block_b2(x)
        x = self.inception_block_b3(x)
        x = self.inception_block_b4(x)
        x = self.inception_block_b5(x)
        x = self.inception_block_c1(x)
        x = self.inception_block_c2(x)
        x = self.inception_block_c3(x)
        x = self.pool3(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.flatten(x)
        return x

def inception(in_channels=1, out_channels=2):
    net = Inceptionv3(in_channels, out_channels)
    return net

if __name__ == "__main__":
    net = inception(1, 2)
    print(net)
    summary(net.cuda(), (1, 100, 272, 256), batch_size=1)