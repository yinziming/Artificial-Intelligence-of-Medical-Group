import torch
import pandas as pd
import torch.nn as nn
import os
import numpy as np
# import vggmodel.resnet as resnet
import densenet
import resnet
import SimpleITK as sitk
from tqdm import tqdm

# net = resnet.generate_model(101)
net = densenet.generate_model(121)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = net.to(device)
data_root = r'F:\all_users\gaohy\data\ddm\croped_data_195'  #
test_weights_path = 'F:/all_users/gaohy/data/ddm/newresult/modelsave/Densenet/all_aug121_epoch-41_val_loss-0.122_val_acc-0.969.pth.tar'  # 预训练模型参数
num_class = 2  # 类别数量
checkpoint = torch.load(test_weights_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
name_list = os.listdir(data_root)
features = pd.DataFrame(index=name_list, columns=list(np.arange(0,1024)))
for path in tqdm(os.listdir(data_root)):
    image = sitk.ReadImage(os.path.join(data_root, path))
    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array[np.newaxis, :]
    input = torch.from_numpy(image_array)
    input = input.unsqueeze(1)
    input = input.to(device, dtype=torch.float)
    feature, output = model(input)
    feature = feature.detach().cpu().numpy()
    feature = np.squeeze(feature)
    features.loc[path, :] = list(feature)
features.to_csv('renji8.csv')

# names = test_dataset.imagelist
# features = pd.DataFrame(index=names)
#
# data = []
# for inputs, labels in test_loader:
#     inputs = inputs.to(device, dtype=torch.float)
#     labels = labels.to(device)
#     feature, outputs = model(inputs)
#     feature = feature.detach().cpu().numpy()
#     data.append(np.squeeze(feature))
#
# for index,name in enumerate(names):
#     features.loc[name] = data[index]
# df = pd.DataFrame(features)
# df.to_csv('feature.csv')
# df = pd.DataFrame(names)
# df.to_csv('names.csv')

