import os
import psutil
import numpy as np
from glob import glob
import SimpleITK as sitk
import shutil
import copy
from tqdm import tqdm
import random

def five_folds_train_valid(crops, croph, cropw):
    # Paths for Brats2018 data_set
    # path_HGG = glob('Res_Unet_torch/data/2019/MICCAI_BraTS_2019_Data_Training/HGG/**')
    # path_LGG = glob('Res_Unet_torch/data/2019/MICCAI_BraTS_2019_Data_Training/LGG/**')
    # path_all = path_HGG + path_LGG  # 拼接两个路径
    path_all = glob(r'Res_Unet_torch\data\2019\dataset/**')

    path_Test = glob(r'Res_Unet_torch\data\2019\test/**')
    path_Test = [each[30:] for each in path_Test]
    path_dataset = path_all

    # for each in path_all:
    #     if each[61:] in path_Test:
    #         continue
    #     else:
    #         path_dataset.append(each)

    np.random.seed(2022)  # 设定随机种子后，一直有效
    np.random.shuffle(path_dataset)  # 随机打乱所有数据的路径

    for i in range(5):
        print(f"making folder {i+1}...")
        start = i * int(len(path_dataset) / 5)
        end = (i + 1) * int(len(path_dataset) / 5) if (i < 4) else None
        path_train = copy.deepcopy(path_dataset)
        path_valid = path_dataset[start:end]
        del path_train[start:end]
        save_path = r"Res_Unet_torch/data/2019/5folders/folder" + str(i+1)
        train_path = os.path.join(save_path, "train")
        for each in path_train:
            file = os.path.join(train_path, each[33:])
            shutil.copytree(each, file)
        # data_preprocess(path_train, train_path, crops, croph, cropw)
        valid_path = os.path.join(save_path, "valid")
        for each in path_valid:
            file = os.path.join(valid_path, each[33:])
            shutil.copytree(each, file)
        # data_preprocess(path_valid, valid_path, crops, croph, cropw)
        print(f"folder {i+1} finish!")

def five_folds_train_valid_test(crops, croph, cropw):
    # Paths for Brats2018 data_set
    path_HGG = glob('Res_Unet_torch/data/2019/MICCAI_BraTS_2019_Data_Training/HGG/**')
    path_LGG = glob('Res_Unet_torch/data/2019/MICCAI_BraTS_2019_Data_Training/LGG/**')
    path_dataset = path_HGG + path_LGG  # 拼接两个路径
    # path_dataset = glob('data\MICCAI_BraTS_2021_Data_Training/**')

    np.random.seed(2022)  # 设定随机种子后，一直有效
    np.random.shuffle(path_dataset)  # 随机打乱所有数据的路径

    for i in range(5):
        print(f"making folder {i+1}...")
        save_path = r"Res_Unet_torch/data/2019/patch_dir/fivefolders_Train_Valid_Test/folder" + str(i+1)
        dst = os.path.join(save_path, "test")
        os.makedirs(save_path)
        start = i * int(len(path_dataset) / 5)
        end = (i + 1) * int(len(path_dataset) / 5) if (i < 4) else None
        path_train = copy.deepcopy(path_dataset)
        path_test = path_dataset[start:end]
        for each in path_test:
            file = os.path.join(dst, each[61:])
            shutil.copytree(each, file)
        del path_train[start:end]
        path_valid = path_train[int(len(path_train) * 4 / 5):]
        train_path = os.path.join(save_path, "train")
        data_preprocess(path_train, train_path, crops, croph, cropw)
        valid_path = os.path.join(save_path, "valid")
        data_preprocess(path_valid, valid_path, crops, croph, cropw)
        print(f"folder {i+1} finish!")

def data_preprocess(base_path, file_path, save_path, crops, croph, cropw, dataset = "train"):
    feature_path = os.path.join(save_path, 'features')
    label_path = os.path.join(save_path, 'labels')
    os.makedirs(feature_path)
    os.makedirs(label_path)
    for i in tqdm(list(range(len(file_path)))):
        flair = glob(base_path + file_path[i] + '/*_flair.nii.gz')
        t2 = glob(base_path + file_path[i] + '/*_t2.nii.gz')
        gt = glob(base_path + file_path[i] + '/*_seg.nii.gz')
        t1 = glob(base_path + file_path[i] + '/*_t1.nii.gz')
        t1c = glob(base_path + file_path[i] + '/*_t1ce.nii.gz')
        t1s = [scan for scan in t1 if scan not in t1c]
        
        if (len(flair)+len(t2)+len(gt)+len(t1s)+len(t1c)) < 5:  # 判断4个模态+标签数据是否加载成功，一次加载一组数据
            print("there is a problem here!!! the problem lies in this patient :", file_path[i])
            continue
        scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
        # read a volume composed of 4 modalities (5,155,240,240)
        # tmp.shape=(5,155,240,240)
        temp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]
        # crop each volume to have a size of (104,128,128) to discard some unwanted background
        
        slices, height, width = temp[0].shape
        strats = slices // 2 - (crops // 2)
        strath = height // 2 - (croph // 2)
        stratw = width // 2 - (cropw // 2)
        temp = np.array(temp)
        train_images = temp[:4, :, :, :]
        features = []
        label = temp[4, :, :, :]
        for train_image in train_images:
            b = np.percentile(train_image, 99)
            t = np.percentile(train_image, 1)
            train_image = np.clip(train_image, t, b)  # 矩阵slice中最小值设置为t=0，最大值设置为b
            image_nonzero = train_image[np.nonzero(train_image)]
            if np.std(train_image)==0 or np.std(image_nonzero) == 0:  # 如果slice标准差为0或非零元素图像的标准差为0，不需要标准化
                continue
            else:
                train_image = (train_image - np.mean(image_nonzero)) / np.std(image_nonzero)
                train_image[train_image == train_image.min()] = -9  # image_nonzero.shape=(240,240)
            features.append(train_image)
        features, label = np.array(features), np.array(label)
        features = features[:, strats:strats+crops, strath:strath+croph, stratw:stratw+cropw]
        label = label[strats:strats+crops, strath:strath+croph, stratw:stratw+cropw]
        # if dataset == "train":
        #     feature_name = feature_path + "/" + file_path[i][33:]
        #     label_name = label_path + "/" + file_path[i][33:] + "label"
        # else:
        #     feature_name = feature_path + "/" + file_path[i][33:]
        #     label_name = label_path + "/" + file_path[i][33:] + "label"

        feature_name = feature_path + "/" + file_path[i]
        label_name = label_path + "/" + file_path[i]
        np.save(feature_name, features)
        np.save(label_name, label)

        del temp
    print("Size of the patches : ", features.shape)
    print("Size of their corresponding targets : ", label.shape)

if __name__ == '__main__':
    #-----------------------------------------2019-----------------------------------
    # five_folds_train_valid(128, 160, 160)
    # five_folds_train_valid_test(128, 160, 160)

    # train_path = glob(r'Res_Unet_torch\data\2019\train\**')
    # valid_path = glob(r'Res_Unet_torch\data\2019\validate\**')
    # save_path_train = r'Res_Unet_torch\data\2019\patch_dir\patch3d128_128_128\train'
    # save_path_valid = r'Res_Unet_torch\data\2019\patch_dir\patch3d128_128_128\valid'
    # data_preprocess(train_path, save_path_train, 128, 128, 128)
    # data_preprocess(valid_path, save_path_valid, 128, 128, 128, dataset="valid")

    #-----------------------------------------2021-----------------------------------
    # 先从2021的数据集中划分出251个数据作为测试集
    path_dataset = os.listdir('data/MICCAI_BraTS2021')
    random.shuffle(path_dataset)
    
    test_set = random.sample(path_dataset, k=121)
    for each in test_set:
        path_dataset.remove(each)
    with open('workspace/Res_Unet_torch/data/2021/test03.txt', 'w') as f:
        for each in test_set:
                f.write(each)
                f.write('\n')
    
    train_path = path_dataset[:1000]
    valid_path = path_dataset[1000:]

    with open('workspace/Res_Unet_torch/data/2021/valid1.txt', 'w') as f:
        for each in valid_path:
                f.write(each)
                f.write('\n')
    with open('workspace/Res_Unet_torch/data/2021/train1.txt', 'w') as f:
        for each in train_path:
                f.write(each)
                f.write('\n')

    # 保存训练集与验证集的数据
    base_path = 'data/MICCAI_BraTS2021/'
    save_path_train = 'workspace/Res_Unet_torch/data/2021/patch_dir/dataset100/train'
    save_path_valid = 'workspace/Res_Unet_torch/data/2021/patch_dir/dataset100/valid'
    data_preprocess(base_path, train_path, save_path_train, 128, 128, 128)
    data_preprocess(base_path, valid_path, save_path_valid, 128, 128, 128, dataset="valid")

    info = psutil.virtual_memory()
    print('内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    print('总内存：', info.total)
    print('内存占比：', info.percent)
    print('cpu个数:', psutil.cpu_count())