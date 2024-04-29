import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchsummary import summary
from thop import profile, clever_format
from copy import deepcopy

class Unet(nn.Module):
    '''
    unet基类, 后续的convnext相关的unet模型全部继承与此基类
    
    args:
        input_layer(nn.module): unet网络的输入层
        encoders(nn.module): unet网络的编码层, 一般为3-4层
        bridge(nn.module): unet网络的bridge结构
        decoders(nn.module): unet网络的解码层, 一般为3-4层, 需要与encode层数对应
        output_layer(nn.module): unet网络的输出层, 输出结果为mask
        istrain(bool): 是否为训练模式, 若为训练模式则会对输入数据增加扰动
    '''
    
    def __init__(self, input_layer, encoders, bridge, decoders, output_layer, istrain = True) -> None:
        super(Unet, self).__init__()
        
        self.input = input_layer
        self.encoder1 = encoders[0]
        self.encoder2 = encoders[1]
        self.encoder3 = encoders[2]
        self.bridge = bridge
        self.decoder1 = decoders[0]
        self.decoder2 = decoders[1]
        self.decoder3 = decoders[2]
        self.output = output_layer
        
        self.istrain = istrain
    
    def forward(self, x):
        if self.istrain:
            x = x + torch.normal(0, 0.01, x.shape, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        x = self.input(x)
        # encoder_out1, x = checkpoint(self.encoder1, x)
        # encoder_out2, x = checkpoint(self.encoder2, x)
        # encoder_out3, x = checkpoint(self.encoder3, x)
        # x = checkpoint(self.bridge, x)
        # x = checkpoint(self.decoder3, encoder_out3, x)
        # x = checkpoint(self.decoder2, encoder_out2, x)
        # x = checkpoint(self.decoder1, encoder_out1, x)
        # x = checkpoint(self.output, x)

        encoder_out1, x = self.encoder1(x)
        encoder_out2, x = self.encoder2(x)
        encoder_out3, x = self.encoder3(x)
        x = self.bridge(x)
        x = self.decoder3(encoder_out3, x)
        x = self.decoder2(encoder_out2, x)
        x = self.decoder1(encoder_out1, x)
        x = self.output(x)
        
        return x

# convnext blocks
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
        
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim_in, dim_out, drop_rate=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dwconv = nn.Conv3d(dim_in, dim_out, kernel_size=7, padding=3, groups=dim_out)  # depthwise conv
        self.norm = LayerNorm(dim_out, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim_out, 4 * dim_out)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim_out, dim_out)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim_out,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # [N, C, S, H, W] -> [N, S, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # [N, S, H, W, C] -> [N, C, S, H, W]

        x = shortcut + self.drop_path(x)
        return x

class LayerNorm2D(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block2D(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim_in, dim_out, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim_in, dim_out, kernel_size=7, padding=3, groups=dim_out)  # depthwise conv
        self.norm = LayerNorm2D(dim_out, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim_out, 4 * dim_out)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim_out, dim_out)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim_out,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, index, dims, depth, drop_rate, cur, layer_scale_init_value) -> None:
        super(EncoderLayer, self).__init__()
        self.stage = nn.Sequential(*[Block(dims[index], dims[index], drop_rate=drop_rate[cur + j], 
                                           layer_scale_init_value=layer_scale_init_value) 
                                     for j in range(depth[index])])
        self.down_sample = nn.Sequential(LayerNorm(dims[index], eps=1e-6, data_format="channels_first"),
                                         nn.Conv3d(dims[index], dims[index+1], kernel_size=2, stride=2))
    
    def forward(self, x):
        encoder_out = self.stage(x)
        x = self.down_sample(encoder_out)
        
        return encoder_out, x

class DecoderLayer(nn.Module):
    def __init__(self, index, dims, depth, upsample_size=None, drop_rate=None, cur=0, layer_scale_init_value=0.) -> None:
        super(DecoderLayer, self).__init__()
        layers = [nn.Conv3d(dims[index]*2, dims[index], kernel_size=1)] + \
                 [Block(dims[index], dims[index], drop_rate=drop_rate[cur+j], layer_scale_init_value=layer_scale_init_value) for j in range(depth[index])]

        self.stage = nn.Sequential(*layers)

        if upsample_size is None:
            self.up_sample = nn.Sequential(LayerNorm(dims[index+1], eps=1e-6, data_format="channels_first"),
                                            nn.ConvTranspose3d(dims[index]*2, dims[index], kernel_size=2, stride=2))
        else:
            self.up_sample = nn.Sequential(LayerNorm(dims[index+1], eps=1e-6, data_format="channels_first"),
                                           nn.Upsample(size=upsample_size[index], mode='trilinear', align_corners=True),
                                           nn.Conv3d(dims[index]*2, dims[index], kernel_size=1))
        
    def forward(self, encoder_out, x):
        x = self.up_sample(x)
        x = torch.cat((x, encoder_out), dim=1)
        x = self.stage(x)
        
        return x

class EncoderLayer2d(nn.Module):
    def __init__(self, index, dims, depth, drop_rate, cur, layer_scale_init_value) -> None:
        super(EncoderLayer2d, self).__init__()
        self.stage = nn.Sequential(*[Block2D(dims[index], dims[index], drop_rate=drop_rate[cur + j], 
                                           layer_scale_init_value=layer_scale_init_value) 
                                     for j in range(depth[index])])
        self.down_sample = nn.Sequential(LayerNorm2D(dims[index], eps=1e-6, data_format="channels_first"),
                                         nn.Conv2d(dims[index], dims[index+1], kernel_size=2, stride=2))
    
    def forward(self, x):
        encoder_out = self.stage(x)
        x = self.down_sample(encoder_out)
        
        return encoder_out, x

class DecoderLayer2d(nn.Module):
    def __init__(self, index, dims, depth, upsample_size=None, drop_rate=None, cur=0, layer_scale_init_value=0.) -> None:
        super(DecoderLayer2d, self).__init__()
        layers = [nn.Conv2d(dims[index]*2, dims[index], kernel_size=1)] + \
                 [Block2D(dims[index], dims[index], drop_rate=drop_rate[cur+j], layer_scale_init_value=layer_scale_init_value) for j in range(depth[index])]

        self.stage = nn.Sequential(*layers)

        if upsample_size is None:
            self.up_sample = nn.Sequential(LayerNorm2D(dims[index+1], eps=1e-6, data_format="channels_first"),
                                            nn.ConvTranspose2d(dims[index]*2, dims[index], kernel_size=2, stride=2))
        else:
            self.up_sample = nn.Sequential(LayerNorm2D(dims[index+1], eps=1e-6, data_format="channels_first"),
                                           nn.Upsample(size=upsample_size[index], mode='trilinear', align_corners=True),
                                           nn.Conv2d(dims[index]*2, dims[index], kernel_size=1))
        
    def forward(self, encoder_out, x):
        x = self.up_sample(x)
        x = torch.cat((x, encoder_out), dim=1)
        x = self.stage(x)
        
        return x

class Unext(Unet):
    '''
    使用convNext作为backbone的unet, 每一个阶段严格按照convNext的block结构
    
    args:
        in_channels(int): 输入通道数
        out_channels(int): 输出通道数
        size_list(list): 图像在上采样后的图像大小 default((115, 240, 240), (38, 60, 60), (19, 30, 30), (9, 15, 15))
        bridge(nn.Module): unext的bridge层
        depth(list): 每一个阶段的conNext的block的个数 default[3, 3, 9, 3]
        dims(list): 每一个阶段的输出通道数 default[96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        istrain(bool): 是否为训练模式, default True
    '''
    def __init__(self, in_channels=4, out_channels=4, size_list=None, depth=None, bridge=None,
                 dims=None, drop_path_rate=0., layer_scale_init_value=1e-6, istrain=True) -> None:
        
        input_layer = nn.Sequential(nn.Conv3d(in_channels, dims[0], kernel_size=4, stride=4),
                                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        
        encoders, decoders = list(), list()
         
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(3):
            encoders.append(EncoderLayer(i, dims, depth, drop_rate=dp_rates, cur=cur, 
                                         layer_scale_init_value=layer_scale_init_value))
            decoders.append(DecoderLayer(i, dims, depth, size_list, drop_rate=dp_rates, cur=cur, 
                                         layer_scale_init_value=layer_scale_init_value))
            cur += depth[i]
        
        bridge_layer = bridge(dims[-1], depth[-1],drop_rate=dp_rates, cur=cur)
        if size_list is None:
            final_upsample = nn.ConvTranspose3d(dims[0], dims[0], kernel_size=4, stride=4)
        else:
            final_upsample = nn.Upsample(size=size_list[-1], mode='trilinear', align_corners=True)
        output_layer = nn.Sequential(final_upsample,
                                     LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                                     nn.Conv3d(dims[0], out_channels, kernel_size=1), nn.Dropout3d(0.),
                                     nn.Softmax(dim=1))
        # output_layer = nn.Sequential(final_upsample,
        #                              LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        #                              nn.Conv3d(dims[0], out_channels, kernel_size=1), nn.Dropout3d(0.2))
        super().__init__(input_layer, encoders, bridge_layer, decoders, output_layer, istrain)

class PrimaryBridge(nn.Module):
    def __init__(self, dim, depth, drop_rate, cur) -> None:
        super(PrimaryBridge, self).__init__()
        self.blk = nn.Sequential(*[Block(dim, dim, drop_rate=drop_rate[cur + j], layer_scale_init_value=1e-6) for j in range(depth)])
    
    def forward(self, x):
        x = self.blk(x)
        return x

class PrimaryBridge2d(nn.Module):
    def __init__(self, dim, depth, drop_rate, cur) -> None:
        super(PrimaryBridge2d, self).__init__()
        self.blk = nn.Sequential(*[Block2D(dim, dim, drop_rate=drop_rate[cur + j], layer_scale_init_value=1e-6) for j in range(depth)])
    
    def forward(self, x):
        x = self.blk(x)
        return x

class Unext_2d(Unet):
    '''
    使用convNext作为backbone的unet, 每一个阶段严格按照convNext的block结构
    
    args:
        in_channels(int): 输入通道数
        out_channels(int): 输出通道数
        size_list(list): 图像在上采样后的图像大小 default((115, 240, 240), (38, 60, 60), (19, 30, 30), (9, 15, 15))
        bridge(nn.Module): unext的bridge层
        depth(list): 每一个阶段的conNext的block的个数 default[3, 3, 9, 3]
        dims(list): 每一个阶段的输出通道数 default[96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        istrain(bool): 是否为训练模式, default True
    '''
    def __init__(self, in_channels=4, out_channels=4, size_list=None, depth=None, bridge=None,
                 dims=None, drop_path_rate=0., layer_scale_init_value=1e-6, istrain=True) -> None:
        
        input_layer = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                                    LayerNorm2D(dims[0], eps=1e-6, data_format="channels_first"))
        
        encoders, decoders = list(), list()
         
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(3):
            encoders.append(EncoderLayer2d(i, dims, depth, drop_rate=dp_rates, cur=cur, 
                                         layer_scale_init_value=layer_scale_init_value))
            decoders.append(DecoderLayer2d(i, dims, depth, size_list, drop_rate=dp_rates, cur=cur, 
                                         layer_scale_init_value=layer_scale_init_value))
            cur += depth[i]
        
        bridge_layer = bridge(dims[-1], depth[-1],drop_rate=dp_rates, cur=cur)
        if size_list is None:
            final_upsample = nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4)
        else:
            final_upsample = nn.Upsample(size=size_list[-1], mode='trilinear', align_corners=True)
        output_layer = nn.Sequential(final_upsample,
                                     LayerNorm2D(dims[0], eps=1e-6, data_format="channels_first"),
                                     nn.Conv2d(dims[0], out_channels, kernel_size=1), nn.Dropout2d(0.),
                                     nn.Softmax2d())
        super().__init__(input_layer, encoders, bridge_layer, decoders, output_layer, istrain)

def unext_tiny(size_list: list, depth=[3, 3, 9, 3], istrain=True):
    bridge = PrimaryBridge
    model = Unext(depth=depth,
                  dims=[96, 192, 384, 768],
                  size_list = size_list, bridge=bridge, drop_path_rate=0., istrain=istrain)
    return model

def unext_tiny2d(in_channels=4, out_channels=4, size_list: list = None, depth=[3, 3, 9, 3], istrain=True):
    bridge = PrimaryBridge2d
    model = Unext_2d(in_channels, out_channels, depth=depth,
                  dims=[96, 192, 384, 768],
                  size_list = size_list, bridge=bridge, drop_path_rate=0., istrain=istrain)
    return model

if __name__ == "__main__":
    size_list = ((128, 128, 128), (64, 64, 64), (32, 32, 32))
    
    net = unext_tiny2d(1, 4, None, depth=[3, 3, 3, 3])
    summary(net.cuda(), (1, 256, 256), batch_size=1)
    flops, params = profile(deepcopy(net.cuda()), inputs=(torch.zeros((1, 1, 256, 256)).cuda(),))
    flops, params = clever_format([flops, params], '%.3f')
    print(f'FLOPs:{flops}, n_params:{params}')

    n = 128
    x = torch.randn(size=(n, 1, 256, 256), requires_grad=True, dtype=torch.float32).cuda()
    torch.cuda.empty_cache()
    while True:
        with torch.autograd.set_detect_anomaly(True):
            _ = net(x)
        torch.cuda.empty_cache()