import ResUnet3D
import unet3d
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
import tqdm
import losses_torch
import os
from torch.nn import functional as F
from torchsummary import summary
import unexts
import lr_scheduler
from thop import profile, clever_format
from copy import deepcopy

class Dataset:
    def __init__(self, file_path, dataset='2019') -> None:
        self.ids = os.listdir(file_path+'/features')
        self.data_dir = file_path
        self.dataset = dataset
        
    def __getitem__(self, i):
        id_ = self.ids[i]
        feature_path = os.path.join(self.data_dir, 'features', id_)
        label_path = os.path.join(self.data_dir, 'labels', id_)
        if self.dataset == "2019":
            # 2019的数据集中label的名称与feature不一致，后面增加了label后缀，所以需要更改
            label_path = os.path.join(self.data_dir, 'labels', id_.split('.')[0]+'label'+'.npy')
        
        feature = np.load(feature_path).astype(np.float32)
        label = np.load(label_path).astype(np.int64)
        
        label[label == 4] = 3
        
        X, y = torch.tensor(feature), torch.tensor(label)
        y = F.one_hot(y, num_classes=4).permute(3,0,1,2)
        
        return X, y
    
    def __len__(self):
        return len(self.ids)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        target = target.type(torch.float32)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.softmax(input, dim=1)
        num = target.size(0)
        input = input.reshape(num, -1)
        target = target.reshape(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class WCE_GDL_SS(nn.Module):
    def __init__(self) -> None:
        super(WCE_GDL_SS, self).__init__()
    
    def forward(self, y_hat, y):
        ss = losses_torch.sensitivity_specificity_loss(y, y_hat)
        wce = losses_torch.weighted_log_loss(y, y_hat)
        gdl = losses_torch.gen_dice_loss(y, y_hat)
        l = gdl + wce + ss
        return l

class WCE_SS(nn.Module):
    def __init__(self) -> None:
        super(WCE_SS, self).__init__()
    
    def forward(self, y_hat, y):
        wce = losses_torch.weighted_log_loss(y, y_hat)
        ss = losses_torch.sensitivity_specificity_loss(y, y_hat)
        l = wce + ss
        return l

class WCE_GDL(nn.Module):
    def __init__(self) -> None:
        super(WCE_GDL, self).__init__()
    
    def forward(self, y_hat, y):
        wce = losses_torch.weighted_log_loss(y, y_hat)
        gdl = losses_torch.gen_dice_loss(y, y_hat)
        l = wce + gdl
        return l

class GDL_SS(nn.Module):
    def __init__(self, alpha = 0.5) -> None:
        super(GDL_SS, self).__init__()
        self.alpha = alpha
    
    def forward(self, y_hat, y):
        gdl = losses_torch.gen_dice_loss(y, y_hat)
        ss = losses_torch.sensitivity_specificity_loss(y, y_hat)
        l = self.alpha * gdl + (1 - self.alpha) * ss
        return l

def train(net, train_iter, valid_iter, num_epochs, loss, optimizer, scheduler, device, logdir, weight_dir, model_dir=None, start_epoch=0, load_weight = False):
    writer = SummaryWriter(log_dir=logdir)
    print("training on", device)
    net.to(device)
    torch.set_float32_matmul_precision('high')
    net = torch.compile(net, backend='inductor')
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
        train_num = 0
        train_loss = 0
        for X, y in tqdm.tqdm(train_iter, 'train epoch:'+str(epoch)):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            with torch.autograd.set_detect_anomaly(True):
                l = loss(y_hat, y)
                l.sum().backward()
            optimizer.step()
            train_num += 1
            train_loss += l.sum()

        train_l = train_loss / train_num
        writer.add_scalar('data/train_loss_epoch', train_l, epoch)
        print("epoch:%d loss:%f " %(epoch, train_l))
        net.eval()
        temp = 0
        valid_num = 0
        for X, y in tqdm.tqdm(valid_iter, 'valid epoch:'+str(epoch)):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                y_hat = net(X)
                l = loss(y_hat, y)
                temp += l.sum()
                valid_num += 1
        val_loss = temp / valid_num
        if (min_val_loss > val_loss) or (epoch == num_epochs-1):
            print(f'valid loss improved from {min_val_loss} to {val_loss}')
            min_val_loss = val_loss
            best_weight_path = os.path.join(weight_dir, 'Unext_epoch-' + str(epoch) + '_loss-%.6f'%val_loss + '.pth.tar')
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, best_weight_path)
            print(f"Save model at {best_weight_path}")
        else:
            print(f'valid loss did not improve!')
        writer.add_scalar('data/val_loss_epoch', val_loss, epoch)
        print("epoch:%d loss:%f" %(epoch, val_loss))

        if scheduler is not None:
            scheduler.step()
            temp = optimizer.param_groups[0]['lr']
            print(f'lr changes to:{temp}')  
        torch.cuda.empty_cache()
    print("Finish")

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv3d:
        nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    finetue = True

    if finetue:
        # 2019 dataset
        train_path = 'workspace/Res_Unet_torch/data/2019/patch_dir/patch3d128_128_128/train'
        valid_path = 'workspace/Res_Unet_torch/data/2019/patch_dir/patch3d128_128_128/valid'
        dataset = '2019'

        # 2019
        batch_size, lr, num_epochs, decay, warmup = 4, 5e-5, 31, 1e-8, 1
        # batch_size, lr, num_epochs, decay, warmup = 1, 1e-3, 150, 1e-4, 1

        logdir = 'workspace/Res_Unet_torch/log/eventsfor2019_unext/recent'
        weight_dir = 'workspace/Res_Unet_torch/pre-trained_weights/2019/'

        load_weight = True
        model_to_load = "workspace/Res_Unet_torch/pre-trained_weights/2021/history/Unext1.0gdl+0.0ss/Unext_epoch-94_loss-0.009236.pth.tar"
    else:
        # 2021 dataset
        train_path = 'workspace/Res_Unet_torch/data/2021/patch_dir/dataset100/train'
        valid_path = 'workspace/Res_Unet_torch/data/2021/patch_dir/dataset100/valid'
        dataset = '2021'

        # 2021
        batch_size, lr, num_epochs, decay, warmup = 4, 5e-3, 300, 1e-4, 20

        logdir = 'workspace/Res_Unet_torch/log/eventsfor2021_unext/recent'    
        weight_dir = 'workspace/Res_Unet_torch/pre-trained_weights/2021'

        load_weight = False
        model_to_load = None

    # size_list for primary
    size_list = ((32, 32, 32), (16, 16, 16), (8, 8, 8), (128, 128, 128))
    net = unexts.unext_tiny(size_list, [3, 3, 3, 3])

    net.apply(xavier_init_weights)
    print(net)
    summary(net.cuda(), (4, 128, 128, 128))
    flops, params = profile(deepcopy(net.cuda()), inputs=(torch.zeros((1, 4, 128, 128, 128)).cuda(),))
    flops, params = clever_format([flops, params], '%.3f')
    print(f'FLOPs:{flops}, n_params:{params}')


    train_set = Dataset(train_path, dataset=dataset)
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_set = Dataset(valid_path, dataset=dataset)
    valid_iter = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=decay, nesterov=False)
    # scheduler = None
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=decay, betas=(0.9, 0.999))
    scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, max_epochs=num_epochs, warmup_epochs=warmup)

    #alpha = [0.1,0.9],step = 0.1
    loss = GDL_SS(1.0)

    train(net, train_iter, valid_iter, num_epochs, loss, optimizer, scheduler, device, 
          logdir, weight_dir, model_dir=model_to_load, start_epoch=0, load_weight=load_weight)
