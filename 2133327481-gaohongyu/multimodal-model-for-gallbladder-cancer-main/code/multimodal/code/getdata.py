import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
import random
import SimpleITK as sitk
from scipy import ndimage
from utils.config import opt

class Dataset:
    '''
    CT图像的dataset类, 供dataloader使用

    args:
        file_path(object): 数据集路径
        istrain(bool): 用于指定数据集, 默认为True, 即训练集, 此时对数据进行数据增强
    '''
    def __init__(self, file_path, istrain=True) -> None:
        self.istrain = istrain
        # 数据集的文件名称都保存在CSV文件中，因此通过读取csv文件的'name'列获取
        dataset_file = pd.read_csv(file_path)
        self.ids = dataset_file['file_path'].values
        self.labels = dataset_file['label'].values
        
    def __getitem__(self, i):
        feature_path = self.ids[i]
        label = self.labels[i]
        # 打开CT图像与相应的label
        feature = sitk.GetArrayFromImage(sitk.ReadImage(feature_path)).astype(np.float32)

        # 图像归一化
        img = self.normalized(feature, 300, 40)

        # 修改图像大小
        img = self.resize_volume(img)

        if self.istrain:
            img = self.enhance(img)
        
        # img.shape=(256, 256,256) -> (1, 256, 256, 256), 增加一维通道维
        img = torch.Tensor(img).unsqueeze(0)
        # label = torch.Tensor([label]).type(torch.float32)

        return img, label
        
    def __len__(self):
        return len(self.ids)
    
    # 300, 40
    def normalized(self, image, ww=300, wl=40):
        '''
        CT图像标准化函数, 将图像的像素值约束在0-1之间

        args:
            image(ndarray): 待归一化的CT图像
            ww(int): 窗宽, 默认为300
            wl(int): 窗位, 默认为40
        
        returns:
            new_image(ndarray): 标准化完成的CT图像

        '''
        upper_gray = wl + 0.5 * ww
        lower_gray = wl - 0.5 * ww
        new_image = (image - lower_gray) / ww
        new_image[new_image < 0] = 0
        new_image[new_image > 1] = 1
        return new_image
    
    def enhance(self, img):
        '''
        ct 图像数据增强函数, 增强策略: 进行随机翻转

        args:
            image(tensor): 待数据增强的CT图像
        
        returns:
            new_image(tensor): 数据增强完成的CT图像
        '''
        if random.random() < 0.5:
            axis = np.random.randint(1, 3)
            new_image = np.flip(img, axis=axis).copy()
        else:
            new_image = img
        
        return new_image
    
    def resize_volume(self, img):
        """修改图像大小"""
        d, w, h = opt.img_size
        # Get current depth
        img_d, img_w, img_h = img.shape
        # Compute depth factor
        depth = img_d / d
        width = img_w / w
        height = img_h / h
        depth_factor =  1 / depth
        width_factor =  1 / width
        height_factor =  1 / height
        img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
        return img

class Multi_Model_Dataset:
    '''
    多模态融合模型的dataset类

    args:
        file_path(object): 数据集路径
        istrain(bool): 用于指定数据集, 默认为True, 即训练集, 此时对数据进行数据增强
    '''
    def __init__(self, opt) -> None:
        ct_path = os.path.join(opt.multi_model_base_path, 'CT')
        ct_malignant_path = os.path.join(ct_path, 'ai', 'data')
        ct_malignant_mask = os.path.join(ct_path, 'ai', 'mask')
        ct_no_cancer_path = os.path.join(ct_path, 'zhengchang', 'data')
        ct_no_cancer_mask = os.path.join(ct_path, 'zhengchang', 'mask')
        self.feature_path = [os.path.join(ct_no_cancer_path, each) for each in os.listdir(ct_no_cancer_path)] + \
                            [os.path.join(ct_malignant_path, each) for each in os.listdir(ct_malignant_path)]
        self.mask_path = [os.path.join(ct_no_cancer_mask, each) for each in os.listdir(ct_no_cancer_mask)] + \
                         [os.path.join(ct_malignant_mask, each) for each in os.listdir(ct_malignant_mask)]
        self.patients = os.listdir(ct_no_cancer_path) + os.listdir(ct_malignant_path)
        self.clinical_data_path = os.path.join(opt.multi_model_base_path, 'linchuang_normalized.csv')
        self.radiomics_data_path = os.path.join(opt.multi_model_base_path, 'fangshe_normalized.csv')
    
    def __getitem__(self, i):
        patient = self.patients[i]
        clinical_data = pd.read_csv(self.clinical_data_path)
        radiomics_data = pd.read_csv(self.radiomics_data_path)

        # 获取label
        label = clinical_data[clinical_data['name'] == patient]['label'].values
        label = torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), num_classes=2).squeeze(0)
        label = label.to(torch.float32)
        # 获取实验室检查数据与影像组学数据
        clinical = clinical_data[clinical_data['name'] == patient].values[0, 2:-1]
        ra = radiomics_data[radiomics_data['fileName'] == patient].values[0, 2:-1]
        clinical = torch.tensor(clinical.astype(np.float32))
        ra = torch.tensor(ra.astype(np.float32))
        # 获取CT
        ct = np.zeros([1,64,64,32])
        feature = sitk.GetArrayFromImage(sitk.ReadImage(self.feature_path[i])).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(self.mask_path[i]))
        img = feature * mask
        img = self.remove_0(img)
        img = self.resize_volume(img)
        ct[0,:]=img[:]  #补齐尺寸
        # ct = self.normalized(ct) #ct归一化
        ct = torch.tensor(ct, dtype=torch.float32)

        return ct, clinical, ra, label
        
    def __len__(self):
        return len(self.patients)
    
    # 300, 40
    def normalized(self, image, ww=300, wl=40):
        '''
        CT图像标准化函数, 将图像的像素值约束在0-1之间

        args:
            image(ndarray): 待归一化的CT图像
            ww(int): 窗宽, 默认为300
            wl(int): 窗位, 默认为40
        
        returns:
            new_image(ndarray): 标准化完成的CT图像

        '''
        upper_gray = wl + 0.5 * ww
        lower_gray = wl - 0.5 * ww
        new_image = (image - lower_gray) / ww
        new_image[new_image < 0] = 0
        new_image[new_image > 1] = 1
        return new_image
    
    def remove_0(self, mat): #切除零边  不然数据量太大
        a = mat.shape
        if np.sum(mat[0,:,:])==0:
            mat = mat[1:,:,:]
        while a[0]!=mat.shape[0]:
            a = mat.shape
            if np.sum(mat[0, :, :]) == 0:
                mat = mat[1:, :, :]

        a = mat.shape
        if np.sum(mat[-1, :, :]) == 0:
            mat = mat[:-1, :, :]
        while a[0] != mat.shape[0]:
            a = mat.shape
            if np.sum(mat[-1, :, :]) == 0:
                mat = mat[:-1, :, :]

        a = mat.shape
        if np.sum(mat[:,0, :]) == 0:
            mat = mat[:,1:, :]
        while a[1] != mat.shape[1]:
            a = mat.shape
            if np.sum(mat[:,0, :]) == 0:
                mat = mat[:,1:, :]
        a = mat.shape
        if np.sum(mat[:,-1, :]) == 0:
            mat = mat[:,:-1, :]
        while a[1] != mat.shape[1]:
            a = mat.shape
            if np.sum(mat[:,-1, :]) == 0:
                mat = mat[:,:-1, :]

        a = mat.shape
        if np.sum(mat[:, :,0]) == 0:
            mat = mat[:, :,1:]
        while a[2] != mat.shape[2]:
            a = mat.shape
            if np.sum(mat[:, :, 0]) == 0:
                mat = mat[:, :, 1:]
        a = mat.shape
        if np.sum(mat[:, :,-1]) == 0:
            mat = mat[:, :,:-1]
        while a[2] != mat.shape[2]:
            a = mat.shape
            if np.sum(mat[:, :, -1]) == 0:
                mat = mat[:, :, :-1]
        return mat
    
    def resize_volume(self, img):
        """修改图像大小"""
        # Set the desired depth
        desired_depth = 32
        desired_width = 64
        desired_height = 64
        # Get current depth
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        # 旋转
        img = ndimage.rotate(img, 90, reshape=False)
        # 数据调整
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img

# for test
if __name__ == '__main__':
    data = Multi_Model_Dataset(opt)
    ct, clinical, ra, label = data.__getitem__(10)
    print(ct.shape, clinical.shape, ra.shape, label.shape)
    train_data,eval_data=random_split(data,
                                  [round(0.8*len(data)),round(0.2*len(data))], #比例在这
                                  generator=torch.Generator().manual_seed(42))  #随机种子
    print(len(train_data), len(eval_data))