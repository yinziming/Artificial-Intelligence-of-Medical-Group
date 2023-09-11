import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from functools import partial
from torchsummary import summary

class EncoderBlock(nn.Module):
    '''
    transformer块

    args:
        num_heads(int): 多头注意力的头数
        hidden_dim(int): 隐藏层输出维度
        mlp_dim(int): transformer 块中mlp层输出的维度
        norm_layer: 归一化层
        dropout(float): dropout 概率, 默认为0.0
        attention_dropout(float): 多头注意力层中的dropout 概率, 默认为0.0
    '''
    def __init__(self, num_heads:int, hidden_dim:int,
                       mlp_dim:int, norm_layer=partial(nn.LayerNorm, eps=1e-6), dropout:float=0.0,
                       attention_dropout:float=0.0) -> None:
        super().__init__()
        self.num_heads = num_heads

        # 多头自注意力层
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # mlp 层
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, hidden_dim), nn.Dropout(dropout))
        
    def forward(self, input:torch.Tensor):
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)

        return x + y

class Encoder(nn.Module):
    '''
    ViT的编码器

    args:
        seq_lenth(int): 序列长度
        num_layers(int): transformer 块的个数
        num_heads(int): 多头注意力的头数
        hidden_dim(int): 隐藏层输出维度
        mlp_dim(int): transformer 块中mlp层输出的维度
        norm_layer: 归一化层
        dropout(float): dropout 概率, 默认为0.0
        attention_dropout(float): 多头注意力层中的dropout 概率, 默认为0.0
    '''
    def __init__(self, seq_lenth:int, num_layers:int, num_heads:int, hidden_dim:int,
                       mlp_dim:int, norm_layer=partial(nn.LayerNorm, eps=1e-6), dropout:float=0.0,
                       attention_dropout:float=0.0) -> None:
        super().__init__()
        # 初始化位置编码
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_lenth, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(num_heads, hidden_dim, mlp_dim, norm_layer,
                                                        dropout, attention_dropout)
        self.layers = nn.Sequential(layers)
        self.norm_layer = norm_layer(hidden_dim)
    
    def forward(self, x:torch.Tensor):
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.layers(x)
        return self.norm_layer(x)

class VisionTransformer(nn.Module):
    '''
    ViT网络结构

    args:
        image_size(int): 输入图像大小, 例如: 图像大小为224*224, 则输入为224
        patch_size(int): 将图像划分为patch的大小
        num_layers(int): transformer 块的个数
        num_heads(int): 多头注意力的头数
        hidden_dim(int): 隐藏层输出维度
        mlp_dim(int): transformer 块中mlp层输出的维度
        dropout(float): dropout 概率, 默认为0.0
        attention_dropout(float): 多头注意力层中的dropout 概率, 默认为0.0
        num_classes(int): 最终层输出的类别个数, 默认为2
        norm_layer: 归一化层
    '''
    def __init__(self, image_size:int, patch_size:int, num_layers:int, num_heads:int,
                       hidden_dim:int, mlp_dim:int, dropout:float=0.0, attention_dropout:float=0.0,
                       num_classes:int=2, norm_layer=partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes

        self.norm_layer = norm_layer(hidden_dim)
        self.conv_proj = nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        # 序列长度
        seq_lenth = (image_size // patch_size) ** 3

        # 增加一个类别token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_lenth += 1

        self.encoder = Encoder(seq_lenth, num_layers, num_heads, hidden_dim,
                               mlp_dim, norm_layer, dropout, attention_dropout)

        self.seq_lenth = seq_lenth

        self.head = nn.Linear(hidden_dim, num_classes)

    def _process_input(self, x:torch.Tensor):
        n, c, d, h, w = x.shape
        p = self.patch_size

        n_d = d // p
        n_h = h // p
        n_w = w // p

        # (n, c, d, h, w)->(n, hidden_dim, n_d, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_d, n_h, n_w)->(n, hidden_dim, n_d*n_h*n_w)
        x = x.reshape(n, self.hidden_dim, n_d*n_h*n_w)
        # (n, hidden_dim, n_d*n_h*n_w)->(n, n_d*n_h*n_w, hidden_dim)
        x = x.permute(0, 2, 1)
        return x
    
    def forward(self, x:torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]

        # (1, 1, hidden_dim)->(n, 1, hidden_dim)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)

        x = x[:, 0]

        x = self.head(x)

        return x

def vit_b_16():
    net = VisionTransformer(image_size=224, 
                            patch_size=16, 
                            num_layers=12, 
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072)
    return net

def vit_b_32():
    net = VisionTransformer(image_size=224, 
                            patch_size=32, 
                            num_layers=12, 
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072)
    return net

def vit_l_16():
    net = VisionTransformer(image_size=224, 
                            patch_size=16, 
                            num_layers=24, 
                            num_heads=16,
                            hidden_dim=1024,
                            mlp_dim=4096)
    return net

def vit_l_32():
    net = VisionTransformer(image_size=224, 
                            patch_size=32, 
                            num_layers=24, 
                            num_heads=16,
                            hidden_dim=1024,
                            mlp_dim=4096)
    return net

def vit_h_14():
    net = VisionTransformer(image_size=224, 
                            patch_size=14, 
                            num_layers=32, 
                            num_heads=16,
                            hidden_dim=1280,
                            mlp_dim=5120)
    return net

if __name__ == "__main__":
    net = vit_l_32()
    print(net)
    x = torch.randn(1, 1, 224, 224, 224).cuda()
    net = net.cuda()
    print(net(x).shape)