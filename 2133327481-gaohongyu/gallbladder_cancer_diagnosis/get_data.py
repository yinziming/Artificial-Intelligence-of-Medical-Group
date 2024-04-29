import os
import pandas as pd
import torch
import SimpleITK as sitk
import random
from utils.config import opt
from torchvision import transforms
import torch
import numpy as np

class Clinical_Dataset():
    def __init__(self, opt, data_set_path, file_path) -> None:
        self.opt = opt
        self.data = pd.read_excel(data_set_path)
        data_set = pd.read_excel(file_path)
        self.patient_name = data_set['id'].values

    def __getitem__(self, i):
        patient = self.patient_name[i]
        # 获取标签
        label = self.data[self.data['id']==patient]['label'].values[0]
        # 获取实验室检查数据
        feature = self.data[self.data['id']==patient].iloc[::, 1:-1].values[0]

        # 对特征进行归一化
        feature = (feature - self.opt.min_num) / (self.opt.max_num - self.opt.min_num)
        feature = torch.tensor(feature, dtype=torch.float32)

        return feature, label

    def __len__(self):
        return len(self.patient_name)

class Single_ct_Dataset():
    def __init__(self, data_path, dataset:str='train_set',use_moco:bool = False) -> None:
        self.data_path = data_path
        if dataset != 'test_set':
            self.dataset = pd.read_csv(os.path.join(data_path, 'dataset', dataset+'.csv'))
        else:
            self.dataset = pd.read_excel('data/gallbladder_detection/dataset/classification/testset/test_set.xlsx')
        self.use_moco = use_moco
        if use_moco:
            self.aug = transforms.Compose([transforms.ToTensor(), 
                                           transforms.Resize((256, 256)),
                                           transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                                           transforms.RandomGrayscale(p=0.2),
                                           transforms.RandomHorizontalFlip()])
        else:
            self.aug = transforms.Compose([transforms.ToTensor(), 
                                           transforms.Resize((224, 224))])
    
    def __getitem__(self, i):
        patient_name = self.dataset.iloc[i]['id']
        feature = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path, patient_name)))
        if self.use_moco:
            img_q = self.aug(feature)
            img_k = self.aug(feature)
            img_q = normalized(img_q)
            img_k = normalized(img_k)
            return [img_q, img_k]
        label = self.dataset.iloc[i]['label']
        img = self.aug(feature)
        img = normalized(img)
        return img, label
    
    def __len__(self):
        return len(self.dataset)
    
class Multi_ct_Dataset():
    def __init__(self, data_path, dataset:str='train_set', n_slice:int=3) -> None:
        self.feature_path = os.path.join(data_path, f'{n_slice}slices')
        self.dataset = pd.read_csv(os.path.join(data_path, 'dataset', dataset+f'_{n_slice}slices.csv'))

    def __getitem__(self, i):
        patient_name = self.dataset.iloc[i]['name']
        feature_names = os.listdir(os.path.join(self.feature_path, patient_name))
        img = torch.Tensor([])
        for slice_name in feature_names:
            feature = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.feature_path, patient_name, slice_name)))
            #feature size: (224, 224)
            feature = normalized(feature)
            feature = torch.FloatTensor(feature)
            # (224, 224) -> (1, 224, 224) -> (1, 1, 224, 224)
            feature = feature.unsqueeze(0)
            feature = feature.unsqueeze(0)
            img = torch.concat([img, feature])

        label = self.dataset.iloc[i]['label']
        return img, label
    
    def __len__(self):
        return len(self.dataset)
    
class KA_ct_Dataset():
    def __init__(self, opt, ct_path, clinical_path, dataset:str='train_set') -> None:
        self.use_multi_slice_encoder = opt.use_multi_slice_encoder
        self.opt = opt
        if opt.use_multi_slice_encoder:
            self.feature_path = os.path.join(ct_path, 'multi_slice',f'{opt.n_slice}slices')
        else:
            self.feature_path = os.path.join(ct_path, 'single_slice')
        if dataset == 'test_set':
            self.dataset = pd.read_excel(os.path.join(clinical_path, f'{dataset}.xlsx'))
        else:
            self.dataset = pd.read_excel(os.path.join(clinical_path, 'dataset.xlsx'))
        self.patient_names = pd.read_excel(os.path.join(clinical_path, f'{dataset}.xlsx'))

        if dataset == 'test_set':
            self.aug = transforms.Compose([transforms.ToTensor()])
        else:
            self.aug = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((256, 256)),
                                           transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                                           transforms.RandomGrayscale(p=0.2),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip()])

    def __getitem__(self, i):
        patient_name = self.patient_names.iloc[i]['id']
        if self.use_multi_slice_encoder:
            feature_names = os.listdir(os.path.join(self.feature_path, patient_name))
            img = torch.Tensor([])
            for slice_name in feature_names:
                feature = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.feature_path, patient_name, slice_name)))
                #feature size: (224, 224)
                feature = normalized(feature).astype(np.float32)
                # feature = torch.FloatTensor(feature)
                feature = self.aug(feature)
                # (224, 224) -> (1, 224, 224) -> (1, 1, 224, 224)
                # feature = feature.unsqueeze(0)
                feature = feature.unsqueeze(0)
                img = torch.concat([img, feature])
        else:
            for each in os.listdir(self.feature_path):
                if patient_name in each and ('rotation' not in each and 'xflip' not in each and 'yflip' not in each):
                    file_name = each
                    break
            feature = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.feature_path,file_name)))
            feature = normalized(feature)
            feature = torch.FloatTensor(feature)
            # (224, 224) -> (1, 224, 224)
            img = feature.unsqueeze(0)
        
        # 获取实验室检查数据
        clinical = self.dataset[self.dataset['id']==patient_name].iloc[::, 1:-1].values[0]

        # 对特征进行归一化
        clinical = (clinical - self.opt.min_num) / (self.opt.max_num - self.opt.min_num)
        clinical = torch.tensor(clinical, dtype=torch.float32)

        label = self.dataset[self.dataset['id']==patient_name].iloc[::, -1].values[0]
        return img, clinical, label
    
    def __len__(self):
        return len(self.patient_names)

# 300, 40
def normalized(image, ww=300, wl=40):
    '''
    CT图像标准化函数, 将图像的像素值约束在0-1之间

    args:
        image(ndarray): 待归一化的CT图像
        ww(int): 窗宽, 默认为300
        wl(int): 窗位, 默认为40
    
    returns:
        new_image(ndarray): 标准化完成的CT图像

    '''
    upper_grey = wl + 0.5 * ww
    lower_grey = wl - 0.5 * ww
    new_image = (image - lower_grey) / ww
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1
    return new_image

if __name__ == '__main__':
    data = Single_ct_Dataset(opt.single_ct_path, 'train_set', True)
    feature, label = data.__getitem__(10)
    print(feature.shape)
    print(label.shape)