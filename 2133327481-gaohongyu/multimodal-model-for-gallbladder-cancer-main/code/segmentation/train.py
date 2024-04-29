import os
from dataset import My_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
# from tensorboardX import SummaryWritertorch
import torch.nn as nn
import numpy as np
import torch
import math
from model import UNet_3D
from loss_function.pytorch_loss_function import DiceLoss
from loss_function.DICE import dice1
import shutil
# from torchvision.utils import make_grid
from common import poly_lr_scheduler

if __name__ == '__main__':
    # data_path = '/home/system/1/sdc/unet+attention/dataall'
    # train_txt_file = '/home/system/1/sdc/unet+attention/dataall/train001.txt'
    # val_txt_file = '/home/system/1/sdc/unet+attention/dataall/val001.txt'
    data_path = '/home/system/1/sdc/unet+attention/original data/fixed_data'
    train_txt_file = '/home/system/1/sdc/unet+attention/original data/fixed_data/train_name_list.txt'
    val_txt_file = '/home/system/1/sdc/unet+attention/original data/fixed_data/val_name_list.txt'

    input_size = (32, 160, 160)
    epoch = 250
    batch_size = 8
    lr_max = 0.0002
    L2 = 0.0001
    cpu = False

    # device = torch.device('cpu')
    device = torch.device('cpu' if cpu else 'cuda')

    save_name = 'bs{}_epoch{}'.format(batch_size, epoch)  ###
    save_dir = os.path.join('new_trained_models', save_name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
    # val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
    print(os.path.join(save_dir))

    train_data = My_Dataset(data_root=data_path, txt_file=train_txt_file, size=input_size)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_data = My_Dataset(data_root=data_path, txt_file=val_txt_file, size=input_size)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print('model loading')
    net = UNet_3D(in_channels=1, out_channels=2, features=(32, 64, 128, 256, 512), norm='batch', act='relu').to(device)  #320 32, 64, 128, 256, 512

    train_data_len = train_data.len
    val_data_len = val_data.len
    print('train_lenth: %i  val_lenth: %i' % (train_data_len, val_data_len))

    Diceloss = DiceLoss(weight=(0.2, 0.8))
    CEloss = nn.CrossEntropyLoss(weight=torch.Tensor((0.2, 0.8)).to(device), reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=L2)

    best_dice = 0

    print('training')
    loss_file_path = '/home/system/1/sdc/unet+attention/fixed_data00/new/loss.txt'
    file_loss = open(loss_file_path, 'w')
    dice_file_path = '/home/system/1/sdc/unet+attention/fixed_data00/new/dice.txt'
    file_dice = open(dice_file_path, 'w')

    for epoch_one in range(epoch):
        poly_lr_scheduler(optimizer, lr_max, epoch_one, lr_decay_iter=1, max_iter=epoch, power=0.9)
        net.train()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        print('lr for this epoch:', lr)
        epoch_train_total_loss = []
        epoch_train_dice = []
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            #print(labels.shape)##############
            #print(torch.unique(labels))###########
            labels_one_hot = torch.zeros((labels.size(0), 2, labels.size(1), labels.size(2), labels.size(3))).to(device). \
                scatter_(1, labels.unsqueeze(1), 1).float()
            results = net(inputs)
            celoss = CEloss(results, labels)
            results = torch.softmax(results, dim=1)
            diceloss = Diceloss(results, labels_one_hot)
            total_loss = 0.6 * diceloss + 0.4 * celoss
            total_loss.backward()
            optimizer.step()
            prediction = torch.argmax(results, dim=1, keepdim=True).cpu().long()
            prediction_one_hot = torch.zeros((prediction.size(0), 2, prediction.size(2), prediction.size(3),
                                              prediction.size(4))).scatter_(1, prediction, 1)
            dice = dice1(prediction_one_hot[:, 1], labels_one_hot[:, 1].cpu()).detach().numpy()
            epoch_train_total_loss.append(total_loss.item())
            epoch_train_dice.append(dice)
            print('[%d/%d, %5d/%d] train_total_loss: %.3f dice: %.3f' % (
                epoch_one + 1, epoch, i + 1, math.ceil(train_data_len / batch_size), total_loss.item(), dice))


        file_loss.write( 'epoch:'+str(epoch_one+1)+'  lr for this epoch:'+ str(lr)+'     ' +'train_total_loss: '+str(total_loss.item())+'      '+ 'dice: ' +str(dice)+'\n')




        with torch.no_grad():
            net.eval()
            epoch_val_total_loss = []
            epoch_val_dice = []
            IMAGE = []
            LABEL = []
            PREDICTION = []
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs, labels = inputs.float().to(device), labels.long().to(device)
                # print(labels.shape)
                # print(torch.unique(labels))
                labels_one_hot = torch.zeros((labels.size(0), 2, labels.size(1), labels.size(2), labels.size(3))).to(device). \
                    scatter_(1, labels.unsqueeze(1), 1).float()
                results = net(inputs)
                celoss = CEloss(results, labels)
                results = torch.softmax(results, dim=1)
                diceloss = Diceloss(results, labels_one_hot)
                total_loss = 0.6 * diceloss + 0.4 * celoss
                prediction = torch.argmax(results, dim=1, keepdim=True).cpu().long()
                prediction_one_hot = torch.zeros((prediction.size(0), 2, prediction.size(2), prediction.size(3),
                                                  prediction.size(4))).scatter_(1, prediction, 1)
                dice = dice1(prediction_one_hot[:, 1], labels_one_hot[:, 1].cpu()).detach().numpy()
                epoch_val_total_loss.append(total_loss.item())
                epoch_val_dice.append(dice)
                if i in [1, 3, 6, 9] and epoch_one % (epoch // 10) == 0:
                    IMAGE.append(inputs[0:1, :, 16, :, :].cpu())
                    LABEL.append(labels[0:1, 16, :, :].cpu().float().unsqueeze(1))
                    PREDICTION.append(prediction[0:1, :, 16, :, :].float())
        epoch_train_total_loss = np.mean(epoch_train_total_loss)
        epoch_train_dice = np.mean(epoch_train_dice)

        epoch_val_total_loss = np.mean(epoch_val_total_loss)
        epoch_val_dice = np.mean(epoch_val_dice)
        print('[%d/%d] train_total_loss: %.3f train_dice: %.3f\nval_total_loss: %.3f val_dice: %.3f' % (
            epoch_one + 1, epoch, epoch_train_total_loss, epoch_train_dice, epoch_val_total_loss, epoch_val_dice))

        file_dice.write( 'epoch:'+str(epoch_one+1)+'  train_total_loss:'+ str(epoch_train_total_loss)+'   ' +'train_dice: '+str(epoch_train_dice)+'    '+ 'val_total_loss: ' +str(epoch_val_total_loss)+'    '+ 'val_dice: ' +str(epoch_val_dice)+'\n')

        # train_writer.add_scalar('lr', lr, epoch_one)
        # train_writer.add_scalar('total_loss', epoch_train_total_loss, epoch_one)
        # train_writer.add_scalar('dice', epoch_train_dice, epoch_one)
        # val_writer.add_scalar('total_loss', epoch_val_total_loss, epoch_one)
        # val_writer.add_scalar('dice', epoch_val_dice, epoch_one)
        # if epoch_one % (epoch // 10) == 0:
        #     IMAGE = torch.cat(IMAGE, dim=0)
        #     LABEL = torch.cat(LABEL, dim=0)
        #     IMAGE = make_grid(IMAGE, 2, normalize=True)
        #     LABEL = make_grid(LABEL, 2, normalize=True)
        #     val_writer.add_image('IMAGE', IMAGE, epoch_one)
        #     val_writer.add_image('LABEL', LABEL, epoch_one)
        #     PREDICTION = torch.cat(PREDICTION, dim=0)
        #     PREDICTION = make_grid(PREDICTION, 2, normalize=True)
        #     val_writer.add_image('PREDICTION', PREDICTION, epoch_one)
        if epoch_one + 1 == epoch:
            torch.save(net.state_dict(),
                       os.path.join(save_dir, 'epoch' + str(epoch_one + 1) + '.pth'))
        if epoch_val_dice > best_dice:
            best_dice = epoch_val_dice
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_dice.pth'))
    # train_writer.close()
    # val_writer.close()
    print('saved_model_name:', save_dir)
    file_loss.close()
    file_dice.close()