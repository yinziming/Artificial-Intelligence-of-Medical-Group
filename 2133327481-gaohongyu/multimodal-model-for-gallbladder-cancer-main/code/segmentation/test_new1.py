import os
from common import norm_img
import shutil
from model import UNet_3D
from Nii_utils import NiiDataWrite, NiiDataRead
from loss_function.pytorch_loss_function import recall_Loss
import math
import torch
import numpy as np

test_data_path = r'/home/system/1/sdc/unet+attention/original data/fixed_data/test'
# model_path = r'/home/system/1/sdc/unet+attention/new/trained_models/best200_dice.pth'
model_path = r'/home/system/1/sdc/unet+attention/fixed_data00/new/new_trained_models/bs8_epoch200/best_dice.pth'
save_path = r'/home/system/1/sdc/unet+attention/original data/fixed_data/testresult'

test_name_list = os.listdir(os.path.join(test_data_path, 'data'))

patch_size = (32, 160, 160)
stride = (16, 80, 80)
cpu = False

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(os.path.join(save_path, 'predictions200'), exist_ok=True)
file = open(os.path.join(save_path, 'evaluation200.txt'), 'w')

device = torch.device('cpu' if cpu else 'cuda')

net = UNet_3D(in_channels=1, out_channels=2, features=(32, 64, 128, 256, 512), norm='batch', act='relu').to(device)

net.load_state_dict(torch.load(model_path))
net.eval()

def compute_dice(pred, label):
    intersection = pred * label
    dice_sco = (2 * intersection.sum()) / (pred.sum() + label.sum())
    return dice_sco

dice_all = []
recall_all = []

with torch.no_grad():
    for i, name in enumerate(test_name_list):
        print(i, name)
        img, spacing, origin, direction = NiiDataRead(
            os.path.join(test_data_path, 'data', name), as_type=np.float32)
        label, _, _, _ = NiiDataRead(
            os.path.join(test_data_path, 'label', name), as_type=np.uint8)
        img = norm_img(img)

        cropped_size = img.shape
        prediction = np.zeros((2, *cropped_size))
        repeat = np.zeros((2, *cropped_size))
        num_z = 1 + math.ceil((cropped_size[0] - patch_size[0]) / stride[0])
        num_x = 1 + math.ceil((cropped_size[1] - patch_size[1]) / stride[1])
        num_y = 1 + math.ceil((cropped_size[2] - patch_size[2]) / stride[2])
        n = 0
        print('original size: {}\ncropped_size: {}\npatches_num: {}*{}*{}'.format(img.shape, cropped_size,
                                                                                  num_z, num_x, num_y))
        total_num = num_z * num_x * num_y
        for z in range(num_z):
            for x in range(num_x):
                for y in range(num_y):
                    print('{}/{}'.format(z * num_x * num_y + x * num_y + y + 1, total_num))
                    x_left = x * stride[1]
                    x_right = x * stride[1] + patch_size[1]
                    y_up = y * stride[2]
                    y_down = y * stride[2] + patch_size[2]
                    z_top = z * stride[0]
                    z_botton = z * stride[0] + patch_size[0]
                    if x == num_x - 1:
                        x_left = cropped_size[1] - patch_size[1]
                        x_right = cropped_size[1]
                    if y == num_y - 1:
                        y_up = cropped_size[2] - patch_size[2]
                        y_down = cropped_size[2]
                    if z == num_z - 1:
                        z_top = cropped_size[0] - patch_size[0]
                        z_botton = cropped_size[0]
                    img_one = np.copy(img[z_top:z_botton, x_left:x_right, y_up:y_down])
                    img_one = torch.from_numpy(img_one).unsqueeze(0).unsqueeze(0).float().to(device)
                    pred_one = net(img_one)
                    pred_one = pred_one.cpu().squeeze(0).detach().numpy()
                    repeat_one = np.ones(pred_one.shape)
                    prediction[:, z_top:z_botton, x_left:x_right, y_up:y_down] += pred_one
                    repeat[:, z_top:z_botton, x_left:x_right, y_up:y_down] += repeat_one
        repeat = repeat.astype(np.float)
        prediction = prediction / repeat
        prediction = np.argmax(prediction, axis=0)

        label1 = np.array(label)
        Label = torch.from_numpy(label1)
        recall_loss = recall_Loss()

        dice = compute_dice((prediction == 1).astype(np.uint8), (label == 1).astype(np.uint8))

        pre = torch.from_numpy(prediction)
        recallloss = recall_loss(pre,Label)

        NiiDataWrite(
            os.path.join(save_path, 'predictions200', name), prediction, spacing, origin, direction, as_type=np.uint8)
        print('{}: {},{}'.format(name, dice, recallloss))
        file.write('{}: {},{}\n'.format(name, dice,recallloss))
        dice_all.append(dice)
        recall_all.append(recallloss)
    dice_all = np.mean(dice_all)
    recall_all = np.mean(recall_all)
    print('mean: {},{}'.format(dice_all,recall_all))
    file.write('mean: {},{}'.format(dice_all,recall_all))
    file.close()