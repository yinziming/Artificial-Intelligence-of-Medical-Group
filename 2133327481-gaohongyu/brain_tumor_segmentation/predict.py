import numpy as np
import pandas as pd
from glob import glob
import os
import SimpleITK as sitk
from evaluation_metrics import *
import torch
from torch import nn
import tqdm
import ResUnet3D
import unet3d
import unexts

def load_test_dataset(file_path, crops, croph, cropw):
    flair = glob(file_path + '/*_flair.nii.gz')
    t2 = glob(file_path + '/*_t2.nii.gz')
    gt = glob(file_path + '/*_seg.nii.gz')
    t1 = glob(file_path + '/*_t1.nii.gz')
    t1c = glob(file_path + '/*_t1ce.nii.gz')
    t1s = [scan for scan in t1 if scan not in t1c]
    
    scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
    temp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]
    temp = np.array(temp).astype(np.float32)
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
            train_image[train_image == train_image.min()] = -9
        features.append(train_image)
    del temp
    features, label = np.array(features), np.array(label).astype(np.uint8)
    features = features[:, strats:strats+crops, strath:strath+croph, stratw:stratw+cropw]
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return features, label

def predict(net, features, device):
    with torch.no_grad():
        X = features.to(device)
        y_hat = net(X)
        # y_hat = nn.Softmax(dim=1)(y_hat)
        result = torch.max(y_hat, 1)[1]
        result = result.cpu().squeeze(0).numpy()
        result[result == 3] = 4
    return result

if __name__ == "__main__":
    test_path = glob('workspace/Res_Unet_torch/data/2019/test/**')
    # 2021 datasets
    # base_path = 'data/MICCAI_BraTS2021/'
    # test_path = [base_path + test_set.strip() for test_set in open('workspace/Res_Unet_torch/data/2021/test.txt')]

    save = False
    show = False
    crops, croph, cropw = 128, 128, 128
    
    model_to_load = "workspace/Res_Unet_torch/pre-trained_weights/2019/Unext_epoch-27_loss-0.009130.pth.tar"  # 加载已训练好的模型
    size_list = ((32, 32, 32), (16, 16, 16), (8, 8, 8), (128, 128, 128))
    net = unexts.unext_tiny(size_list, [3, 3, 3, 3], False)

    # 3d resunet config
    # crops, croph, cropw = 155, 240, 240
    # size_list = ((38, 60, 60), (77, 120, 120), (155, 240, 240))
    # net = ResUnet3D.ResUnet(4, 4, is_train=False, size_list=size_list)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    checkpoint = torch.load(model_to_load)
    # 使用compile优化网络后在推理阶段也需要写
    torch.set_float32_matmul_precision('high')
    net = torch.compile(net, backend='inductor')
    net.load_state_dict(checkpoint['state_dict'])

    net.eval()
    results = []
    count = 0
    low_counter = 0
    low_name = []
    for i in tqdm.tqdm(range(len(test_path))):
        save_path = os.path.basename(test_path[i])
        features, labels = load_test_dataset(test_path[i], crops, croph, cropw)
        preds = predict(net, features, device)

        strats = 155 // 2 - (crops // 2)
        strath = 240 // 2 - (croph // 2)
        stratw = 240 // 2 - (cropw // 2)
        feature_pred = np.zeros((155, 240, 240), dtype=preds.dtype)
        feature_pred[strats:strats+crops, strath:strath+croph, stratw:stratw+cropw] = preds

        if save:
            feature_pred = feature_pred.astype(np.int16)
            tmp=sitk.GetImageFromArray(feature_pred)
            sitk.WriteImage (tmp,'workspace/Res_Unet_torch/predict/2019/ct/dropout0.2/{}.nii.gz'.format(save_path))

        Dice_complete = DSC_whole(feature_pred,labels)
        Dice_enhancing = DSC_en(feature_pred,labels)
        Dice_core = DSC_core(feature_pred,labels)

        Sensitivity_whole = sensitivity_whole(feature_pred, labels)
        Sensitivity_en = sensitivity_en(feature_pred, labels)
        Sensitivity_core = sensitivity_core(feature_pred, labels)
        
        Specificity_whole = specificity_whole(feature_pred, labels)
        Specificity_en = specificity_en(feature_pred, labels)
        Specificity_core=specificity_core(feature_pred, labels)

        Hausdorff_whole=hausdorff_whole(feature_pred, labels)
        Hausdorff_en=hausdorff_en(feature_pred, labels)
        Hausdorff_core=hausdorff_core(feature_pred, labels)

        result = [test_path[i][30:], Dice_complete,Dice_core,Dice_enhancing,Sensitivity_whole,Sensitivity_core,Sensitivity_en,Specificity_whole,Specificity_core,Specificity_en,Hausdorff_whole,Hausdorff_core,Hausdorff_en]
        # result = [test_path[i][41:], Dice_complete,Dice_core,Dice_enhancing,Sensitivity_whole,Sensitivity_core,Sensitivity_en,Specificity_whole,Specificity_core,Specificity_en,Hausdorff_whole,Hausdorff_core,Hausdorff_en]
        if show:
            print("************************************************************")
            print("Dice complete tumor score : {:0.4f}".format(Dice_complete))
            print("Dice core tumor score (tt sauf vert): {:0.4f}".format(Dice_core))
            print("Dice enhancing tumor score (jaune):{:0.4f} ".format(Dice_enhancing))
            print("**********************************************")
            print("Sensitivity complete tumor score : {:0.4f}".format(Sensitivity_whole))
            print("Sensitivity core tumor score (tt sauf vert): {:0.4f}".format(Sensitivity_core))
            print("Sensitivity enhancing tumor score (jaune):{:0.4f} ".format(Sensitivity_en))
            print("***********************************************")
            print("Specificity complete tumor score : {:0.4f}".format(Specificity_whole))
            print("Specificity core tumor score (tt sauf vert): {:0.4f}".format(Specificity_core))
            print("Specificity enhancing tumor score (jaune):{:0.4f} ".format(Specificity_en))
            print("***********************************************")
            print("Hausdorff complete tumor score : {:0.4f}".format(Hausdorff_whole))
            print("Hausdorff core tumor score (tt sauf vert): {:0.4f}".format(Hausdorff_core))
            print("Hausdorff enhancing tumor score (jaune):{:0.4f} ".format(Hausdorff_en))
            print("***************************************************************\n\n")

        if Dice_enhancing <= 0.75:
            low_counter += 1
            low_name.append(test_path[i][22:])
        count += 1
        results.append(result)

    temp=np.array([result[1:] for result in results])
    results=np.array(results)
    columns = ['file', 'dice_wt', 'dice_tc', 'dice_et', 'Sensitivity_wt', 'Sensitivity_tc', 'Sensitivity_et', 'Specificity_wt', 'Specificity_tc', 'Specificity_et', 'hausfroff_wt', 'hausdroff_tc', 'hausdroff_et']
    df = pd.DataFrame(results, columns=columns)
    print("mean : ",np.mean(temp,axis=0))
    print("std : ",np.std(temp,axis=0))
    print("median : ",np.median(temp,axis=0))
    print("25 quantile : ",np.percentile(temp,25,axis=0))
    print("75 quantile : ",np.percentile(temp,75,axis=0))
    print("max : ",np.max(temp,axis=0))
    print("min : ",np.min(temp,axis=0))
    print("count:", count)

    print("low:", low_counter)
    for each in low_name:
        print(each)
    # np.savetxt(r'Res_Unet_torch\predict\2019\Results.txt', results, fmt='%s')
    # df.to_csv('workspace/Res_Unet_torch/predict/2021/Results_01.csv', index=False)
