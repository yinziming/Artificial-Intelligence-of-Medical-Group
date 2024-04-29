import os
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from models.getnet import get_net
from torchsummary import summary
import getdata
from utils.config import opt

def train(net, train_iter, valid_iter, opt, num_epochs=0, loss=None, optimizer=None, device=None, 
          log_dir=None, weight_dir=None, model_dir=None, start_epoch=0, load_weight = False):
    '''
    胆囊癌分类模型训练函数

    args:
        net(nn.Module): 神经网络
        train_iter(DataLoader): 训练数据
        valid_iter(DataLoader): 验证数据
        opt(Object): 配置文件中的参数信息
        num_epochs(int): 训练轮数
        loss(nn.Module): loss函数, 用于计算loss
        optimizer(torch.optim): 优化器, 用于优化网络模型
        device(torch.device): 网络训练的硬件环境(cpu or cuda)
        logdir(str): 网络训练信息保存路径
        weight_dir(str): 网络训练时的权重保存路径
        model_dir(str): 网络训练时的权重载入路径, 用于'断点续训练'
        start_epoch(int): 训练开始轮数, 用于'断点续训练'
        load_weight(bool): 是否载入权重, 用于'断点续训练'
    '''
    # 设置权重保存路径与log信息保存路径
    weight_dir = os.path.join(weight_dir, opt.net_name)
    log_dir = os.path.join(log_dir, opt.net_name)

    # 若没有此路径则先创建路径
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # 设置log信息写入器
    writer = SummaryWriter(log_dir=log_dir)
    print("training on", device)
    net.to(device)
    if load_weight:
        checkpoint = torch.load(model_dir)
        net.load_state_dict(checkpoint['state_dict'])
        print("load model success!")
    print(f'training from epoch{start_epoch}')

    min_val_loss = torch.inf
    base_lr = optimizer.param_groups[0]['lr']
    print(f'lr:{base_lr}')

    for epoch in range(start_epoch, num_epochs):
        net.train()
        train_l = 0
        train_acc = 0
        train_num = 0
        train_loss = 0
        n_y = 0
        for X, y in tqdm.tqdm(train_iter, 'train epoch:'+str(epoch)):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            with torch.autograd.set_detect_anomaly(True):
                l = loss(y_hat, y)
                l.backward()
            optimizer.step()
            train_num += 1
            train_loss += l
            n_y += y.shape[0]

            y_hat = nn.Softmax(dim=1)(y_hat)
            pred = torch.argmax(y_hat, dim=1)
            acc = (pred == y).sum()
            train_acc += float(acc)

        train_l = train_loss / train_num
        train_acc /= n_y

        # 记录训练信息
        writer.add_scalar('data/train_loss_epoch', train_l, epoch)
        writer.add_scalar('data/train_acc_epoch', train_acc, epoch)
        print("epoch:%d loss:%f acc:%f" %(epoch, train_l, train_acc))
        net.eval()

        # 网络验证
        valid_loss = 0
        valid_num = 0
        valid_acc = 0
        n_y = 0
        for X, y in tqdm.tqdm(valid_iter, 'valid epoch:'+str(epoch)):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                y_hat = net(X)
                l = loss(y_hat, y)
                valid_loss += l
                valid_num += 1
                n_y += y.shape[0]
                
                y_hat = nn.Softmax(dim=1)(y_hat)
                pred = torch.argmax(y_hat, dim=1)
                acc = (pred == y).sum()
                valid_acc += float(acc)
        valid_loss /= valid_num
        valid_acc /= n_y

        # 记录验证集的信息
        writer.add_scalar('data/val_loss_epoch', valid_loss, epoch)
        writer.add_scalar('data/val_loss_epoch', valid_acc, epoch)
        print("epoch:%d loss:%f acc:%f" %(epoch, valid_loss, valid_acc))

        # 若当前的valid loss比最低的loss低,则保存当前权重
        if (min_val_loss > valid_loss):
            print(f'valid loss improved from {min_val_loss} to {valid_loss}')
            min_val_loss = valid_loss
            best_weight_path = os.path.join(weight_dir, opt.net_name + '_epoch-' + str(epoch) + '_loss-%.6f'%valid_loss + '_acc-%.6f'%valid_acc + '.pth.tar')
            torch.save({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, best_weight_path)
            print(f"Save model at {best_weight_path}")
        else:
            print(f'valid loss did not improve!')

        torch.cuda.empty_cache()
    print("Finish")

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv3d:
        nn.init.xavier_uniform_(m.weight)

if __name__ == "__main__":
    # 获取数据集
    train_set = getdata.Dataset(opt.train_path, istrain=True)
    valid_set = getdata.Dataset(opt.valid_path, istrain=False)
    train_iter = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    valid_iter = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=False)

    # 载入网络
    size_list = None
    net = get_net(opt.net_name)
    net.apply(xavier_init_weights)
    print(net)
    summary(net.cuda(), (1, 256, 256, 256))

    # 设置优化方式
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay, nesterov=False)
    
    # 设置loss函数
    loss = torch.nn.CrossEntropyLoss()

    # 开始训练
    train(net, train_iter, valid_iter, opt, opt.epoch, loss, optimizer, opt.device, 
          opt.log_dir, opt.weight_dir, model_dir=opt.weight_to_load, start_epoch=opt.trained_epoch, 
          load_weight=opt.load_weight)