from models.convnext import *
from models.inception import *
from models.mobilenet import *
from models.vit import *


def get_net(net_name:str):
    '''
    生成网络函数, 根据传入的参数生成相应的网络

    args:
        net_name(str): 所要生成的网络名称

    returns:
        net(nn.Module): 所生成的网络
    '''
    assert net_name in ['Convnext_T', 'Convnext_S', 'Convnext_B', 'Mobilenet', 'InceptionV3', 
                        'ViT-B-16', 'ViT-B-32', 'ViT-L-16', 'ViT-L-32', 'ViT-H-14']
    if net_name == 'Convnext_T':
        net = convnext_tiny()
    if net_name == 'Convnext_S':
        net = convnext_small()
    if net_name == 'Convnext_B':
        net = convnext_base()
    if net_name == 'Convnext_L':
        net = convnext_large()
    if net_name == 'Convnext_X':
        net = convnext_xlarge()
    if net_name == 'Mobilenet':
        net = mobile_net()
    if net_name == 'InceptionV3':
        net = inception()
    if net_name == 'ViT-B-16':
        net = vit_b_16()
    if net_name == 'ViT-B-32':
        net = vit_b_32()
    if net_name == 'ViT-L-16':
        net = vit_l_16()
    if net_name == 'ViT-L-32':
        net = vit_l_32()
    if net_name == 'ViT-H-14':
        net = vit_h_14()
    return net