import torch
from torch import nn
import torchvision
import models.moco
import models.multi_slice
from models.mlp import Clinical_Classificator
from models.ka_fusion import KA_Module

def get_net(net_name:str, weight_save_path:str=None, n_classes:int=3, use_moco:bool=False):
    '''
    模型获取函数

    args:
        net_name(str): 网路名称
        pretrained(bool): 是否加载预训练权重
        weight_save_path(bool): 预训练权重保存路径
        n_classes(int): 网络输出类别
    
    returns:
        net(nn.Module): 网络模型
    '''
    if net_name == 'resnet_18':
        net = torchvision.models.resnet18()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.fc = nn.Linear(net_weights['fc.weight'].shape[1], n_classes)
    
    if net_name == 'resnet_34':
        net = torchvision.models.resnet34()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.fc = nn.Linear(net_weights['fc.weight'].shape[1], n_classes)

    if net_name == 'resnet_50':
        net = torchvision.models.resnet50()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.fc = nn.Linear(net_weights['fc.weight'].shape[1], n_classes)
    
    if net_name == 'resnet_101':
        net = torchvision.models.resnet101()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.fc = nn.Linear(net_weights['fc.weight'].shape[1], n_classes)
    
    if net_name == 'resnet_152':
        net = torchvision.models.resnet152()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.fc = nn.Linear(net_weights['fc.weight'].shape[1], n_classes)
    
    if net_name == 'resnext_50_32x4d':
        net = torchvision.models.resnext50_32x4d()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.fc = nn.Linear(net_weights['fc.weight'].shape[1], n_classes)
    
    if net_name == 'resnext_101_64x4d':
        net = torchvision.models.resnext101_64x4d()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.fc = nn.Linear(net_weights['fc.weight'].shape[1], n_classes)
    
    if net_name == 'resnext_101_32x8d':
        net = torchvision.models.resnext101_32x8d()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.fc = nn.Linear(net_weights['fc.weight'].shape[1], n_classes)
    
    if net_name == 'densenet_121':
        net = torchvision.models.densenet121()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.classifier = nn.Linear(net.classifier.weight.shape[1], n_classes)
    
    if net_name == 'densenet_161':
        net = torchvision.models.densenet161()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.classifier = nn.Linear(net.classifier.weight.shape[1], n_classes)
    
    if net_name == 'densenet_169':
        net = torchvision.models.densenet169()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.classifier = nn.Linear(net.classifier.weight.shape[1], n_classes)
    
    if net_name == 'densenet_201':
        net = torchvision.models.densenet201()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.classifier = nn.Linear(net.classifier.weight.shape[1], n_classes)
    
    if net_name == 'convnext_T':
        net = torchvision.models.convnext_tiny()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.features[0][0] = nn.Conv2d(1, net.features[0][0].weight.shape[0], kernel_size=(4, 4), stride=(4, 4), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.classifier[2] = nn.Linear(net.classifier[2].weight.shape[1], n_classes)
    
    if net_name == 'convnext_S':
        net = torchvision.models.convnext_small()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.classifier[2] = nn.Linear(net.classifier[2].weight.shape[1], n_classes)
    
    if net_name == 'convnext_B':
        net = torchvision.models.convnext_base()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.features[0][0] = nn.Conv2d(1, net.features[0][0].weight.shape[0], kernel_size=(4, 4), stride=(4, 4), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.classifier[2] = nn.Linear(net.classifier[2].weight.shape[1], n_classes)
    
    if net_name == 'convnext_L':
        net = torchvision.models.convnext_large()
        # 预训练好的模型输入层为3通道,因此需要将输入层通道更改为1通道,将原卷积层预训练的权重在第一维求平均作为新卷积层的权重
        net_weights = net.state_dict()
        net.features[0][0] = nn.Conv2d(1, net.features[0][0].weight.shape[0], kernel_size=(4, 4), stride=(4, 4), bias=False)
        if weight_save_path:
            if use_moco:
                # 将除全连接层以外的所有层的参数替换为moco的预训练权重
                moco_weights = torch.load(weight_save_path)['state dict']
                weights_name_arch = list(net_weights.keys())[:-2]
                weights_name_moco = list(moco_weights.keys())[:-4]
                for i in range(len(weights_name_arch)):
                    net_weights[weights_name_arch[i]] = moco_weights[weights_name_moco[i]]
                net.load_state_dict(net_weights)
            # 将网络的fc层替换为输出通道为n_classes的fc层
        net.classifier[2] = nn.Linear(net.classifier[2].weight.shape[1], n_classes)

    return net

def get_moco(opt):
    '''
    moco模型获取函数,用于预训练moco模型
    
    args:
        opt(object): 配置参数

    returns:
        net(nn.Module):moco模型
    '''

    if 'resnet' in opt.backbone_name:
        base_net = get_net(opt.backbone_name, n_classes=opt.dim)
        dim_mlp = base_net.fc.weight.shape[1]
        base_net.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), base_net.fc)
        net = models.moco.MoCo(base_net, opt.dim, opt.K, opt.m, opt.T)

    if 'resnext' in opt.backbone_name:
        base_net = get_net(opt.backbone_name, n_classes=opt.dim)
        dim_mlp = base_net.fc.weight.shape[1]
        base_net.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), base_net.fc)
        net = models.moco.MoCo(base_net, opt.dim, opt.K, opt.m, opt.T)
    
    if 'densenet' in opt.backbone_name:
        base_net = get_net(opt.backbone_name, n_classes=opt.dim)
        dim_mlp = base_net.classifier.weight.shape[1]
        base_net.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), base_net.classifier)
        net = models.moco.MoCo(base_net, opt.dim, opt.K, opt.m, opt.T)
    
    if 'convnext' in opt.backbone_name:
        base_net = get_net(opt.backbone_name, n_classes=opt.dim)
        dim_mlp = base_net.classifier[2].weight.shape[1]
        base_net.classifier = nn.Sequential(nn.Flatten(), nn.LayerNorm(dim_mlp, eps=1e-6), nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, opt.dim))
        net = models.moco.MoCo(base_net, opt.dim, opt.K, opt.m, opt.T)
    return net

def get_multi_slice_model(opt, load_weights:bool=False):
    '''
    多ct联合诊断模型获取函数,用于预训练moco模型
    
    args:
        opt(object): 配置参数
        load_weights(bool): 是否载入预训练权重, 模型预测用

    returns:
        net(nn.Module):多ct联合诊断模型
    '''

    hidden_size = {'resnet_18':512, 'resnet_34':512, 'resnet_50':2048,
                   'resnet_101':2048, 'resnet_152':2048,
                   'resnext_50_32x4d':2048, 'resnext_101_32x8d':2048, 'resnext_101_64x4d':2048,
                   'densenet_121':1024, 'densenet_161':2208,
                   'densenet_169':1664, 'densenet_201':1920,
                   'convnext_T':768, 'convnext_S':768, 'convnext_B':1024, 'convnext_T':1536}
    base_net = get_net(opt.backbone_name, n_classes=opt.n_class, use_moco=False)
    if not load_weights:
        weights = torch.load(opt.single_ct_saved_weights_path)['state dict']
        base_net.load_state_dict(weights)
    # 去掉网络最后一层，让网络直接输出池化层之后的结果
    if 'resnet' in opt.backbone_name:
        base_net.fc = nn.Sequential(*[])
    if 'resnext' in opt.backbone_name:
        base_net.fc = nn.Sequential(*[])
    if 'densenet' in opt.backbone_name:
        base_net.classifier = nn.Sequential(*[])
    if 'convnext' in opt.backbone_name:
        base_net.classifier[2] = nn.Sequential(*[])
    
    net = models.multi_slice.Multi_slice_Model(encoder=base_net, hidden_size=hidden_size[opt.backbone_name],
                                               n_classes=opt.n_class, fusion_mode=opt.fusion_mode)
    if load_weights:
        weights = torch.load(opt.multi_ct_saved_weights_path)['state dict']
        net.load_state_dict(weights)

    return net

def get_mlp(opt, load_weight:bool=False):
    net = Clinical_Classificator(opt.num_inputs, opt.num_hidden, opt.n_class)
    if load_weight:
        weights = torch.load(opt.clinical_weight_to_load)['state dict']
        net.load_state_dict(weights)

    return net

def get_ka_model(opt, load_weights:bool=False):
    ct_hidden_size = {'resnet_18':512, 'resnet_34':512, 'resnet_50':2048,
                   'resnet_101':2048, 'resnet_152':2048,
                   'resnext_50_32x4d':2048, 'resnext_101_32x8d':2048, 'resnext_101_64x4d':2048,
                   'densenet_121':1024, 'densenet_161':2208,
                   'densenet_169':1664, 'densenet_201':1920,
                   'convnext_T':768, 'convnext_S':768, 'convnext_B':1024, 'convnext_T':1536}
    
    if opt.use_multi_slice_encoder:
        ct_encoder = get_multi_slice_model(opt, True)
        ct_encoder.fc = nn.Sequential(*[])
    else:
        ct_encoder = get_net(opt.backbone_name, n_classes=opt.n_class)
        weights = torch.load(opt.single_ct_saved_weights_path)['state dict']
        ct_encoder.load_state_dict(weights)
        # 去掉网络最后一层，让网络直接输出池化层之后的结果
        if 'resnet' in opt.backbone_name:
            ct_encoder.fc = nn.Sequential(*[])
        if 'resnext' in opt.backbone_name:
            ct_encoder.fc = nn.Sequential(*[])
        if 'densenet' in opt.backbone_name:
            ct_encoder.classifier = nn.Sequential(*[])
        if 'convnext' in opt.backbone_name:
            ct_encoder.classifier[2] = nn.Sequential(*[])
    
    clinical_encoder = get_mlp(opt, True)
    clinical_encoder.fc = nn.Sequential(*[])

    for p in ct_encoder.parameters():
        p.requires_grad = False
    for p in clinical_encoder.parameters():
        p.requires_grad = False

    net = KA_Module(ct_encoder, clinical_encoder, ct_hidden_size[opt.backbone_name],
                    opt.num_hidden, opt.ka_hidden_dim, opt.n_class, opt.KA_fusion_mode)
    
    if load_weights:
        weights = torch.load(opt.KA_saved_weights_path)['state dict']
        net.load_state_dict(weights)

    return net
    

if __name__ == "__main__":
    net = get_net('resnet_50')
    net_weights = net.state_dict()
    print(net_weights.keys())