import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.utils.data.distributed
from get_data import Single_ct_Dataset
from models.get_net import get_moco
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.config import opt
import math

def train(net:nn.Module, train_iter:DataLoader, opt:object, loss:nn, optimizer:torch.optim):
    '''
    实验室检查数据分类模型训练函数

    args:
        net(nn.Module): 待训练的网络
        train_iter(DataLoader): 训练集
        valid_iter(DataLoader): 验证集
        opt(object): 配置文件中的参数信息
        loss(nn): 损失函数
        optimizer(torch.optim):优化器
    '''

    # 创建权重保存路径与log信息保存路径
    weight_dir = os.path.join(opt.moco_weight_dir, opt.backbone_name)
    log_dir = os.path.join(opt.moco_log_dir, opt.backbone_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置log信息写入器
    writer = SummaryWriter(log_dir=log_dir)

    print('Training on', opt.device)
    net.to(opt.device)

    min_val_loss = torch.inf

    for epoch in range(opt.moco_epoch):
        adjust_learning_rate(optim, epoch, opt)
        net.train()
        train_loss, train_num = 0, 0
        for img_q, img_k in tqdm(train_iter, 'train epoch:'+str(epoch)):
            optimizer.zero_grad()
            img_q, img_k = img_q.to(opt.device), img_k.to(opt.device)
            outputs, target = net(img_q, img_k)
            l = loss(outputs, target)
            l.backward()
            optimizer.step()

            train_num += 1
            train_loss += l
        train_loss /= train_num

        # 记录训练信息
        writer.add_scalar('data/train_loss_epoch', train_loss, epoch)
        print('epoch:%d loss:%f' % (epoch, train_loss))

        # 若当前的alid loss 比最低的loss低, 则保存当前权重
        if min_val_loss > train_loss:
            print(f'valid loss imporved from {min_val_loss} to {train_loss}')
            min_val_loss = train_loss
            best_weight_dir = os.path.join(weight_dir, 'moco_v2' + opt.backbone_name + '.pth.tar')
            torch.save({'state dict' : net.state_dict()}, best_weight_dir)
            print(f'save model at {best_weight_dir}')
        else:
            print(f'valid loss did not improve! The lowest loss is {min_val_loss}')
        
        torch.cuda.empty_cache()

    print(f'finish!, valid loss {min_val_loss}')

    return best_weight_dir

def adjust_learning_rate(optimizer, epoch, opt):
    """Decay the learning rate based on schedule"""
    lr = opt.moco_lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / opt.moco_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    train_set = Single_ct_Dataset(opt.single_ct_path, 'train_moco_set', True)
    train_iter = DataLoader(train_set, batch_size=opt.moco_batch_size, shuffle=True, drop_last=True)

    net = get_moco(opt)
    print(net)
    print(f'current backbone:{opt.backbone_name}')
    net.cuda()
    
    loss = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(net.parameters(), lr=opt.moco_lr, weight_decay=opt.weight_decay, momentum=0.9)

    weight_dir = train(net, train_iter, opt, loss, optim)
    torch.cuda.empty_cache()

    # 提取moco预训练模型中encoder_q的权重
    weights = torch.load(weight_dir)
    new_dict = {}
    for key in weights['state dict'].keys():
        if 'encoder_q.' in key:
            new_dict[key] = weights['state dict'][key]
    torch.save({'state dict': new_dict}, weight_dir)