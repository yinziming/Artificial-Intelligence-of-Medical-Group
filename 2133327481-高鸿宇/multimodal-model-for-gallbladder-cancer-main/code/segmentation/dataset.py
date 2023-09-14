import numpy as np
import os
from Nii_utils import NiiDataRead
from monai.transforms import Compose, Rand3DElasticd, SpatialPadd, RandAdjustContrastd, RandGaussianNoised, ToTensord, RandSpatialCropd, RandFlipd
from torch.utils.data import DataLoader, Dataset
from common import norm_img

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list

class My_Dataset(Dataset):
    def __init__(self, data_root='data', txt_file='', size=(32, 128, 128), mode='train'):
        self.data_root = data_root
        self.file_list = load_file_name_list(txt_file)
        if mode == 'train':
            self.transforms = Compose([
                SpatialPadd(keys=["img", "seg"], spatial_size=size),##不符合输出尺寸的进行填充
                RandSpatialCropd(keys=["img", "seg"], roi_size=size, random_size=False),###随机裁剪
                Rand3DElasticd(keys=["img", "seg"], sigma_range=(0, 2), magnitude_range=(0, 2), spatial_size=size,
                               rotate_range=(np.pi / 360 * 50, np.pi / 360 * 10, np.pi / 360 * 10),
                               scale_range=(0.2, 0.2, 0.2),
                               padding_mode="zeros",
                               mode='nearest',
                               prob=0.3),  ###随机3D弹性变形
                RandFlipd(keys=["img", "seg"], prob=0.3, spatial_axis=[0, 1, 2]),   ##图像随机翻转
                # RandAdjustContrastd(keys="img", prob=0.3, gamma=(0.7, 2)),
                # RandGaussianNoised(keys="img", prob=0.3, mean=0.0, std=0.03),
                ToTensord(keys=["img", "seg"])
            ])
        else:
            self.transforms = Compose([
                SpatialPadd(keys=["img", "seg"], spatial_size=size),
                RandSpatialCropd(keys=["img", "seg"], roi_size=size, random_size=False),
                ToTensord(keys=["img", "seg"])
            ])
        self.len = len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        img, _, _, _ = NiiDataRead(os.path.join(self.data_root, 'data', name))
        label, _, _, _ = NiiDataRead(os.path.join(self.data_root, 'label', name))
        img = norm_img(img)

        img = img[np.newaxis, ...]
        label = label[np.newaxis, ...]
        augmented = self.transforms({'img': img, 'seg': label})
        img = augmented['img'].float()
        label = augmented['seg'][0].long()
        return img, label

    def __len__(self):
        return self.len
