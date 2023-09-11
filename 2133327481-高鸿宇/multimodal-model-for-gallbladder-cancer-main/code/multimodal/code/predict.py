import os
import pandas as pd
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from models.getnet import get_net
import getdata
from utils.config import opt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def predict(net, test_iter, opt):
    '''
    CT 分类网络推理函数, 用于将训练好的网络在测试集上进行推理, 验证网络的性能

    args:
        net(nn.Module): 训练好的模型
        test_iter(dataloder): 测试集数据
        opt(Object): 配置文件中的参数信息
    
    returns:
        result(ndarray): 推理结果, 形状大小为(n, 2), 第1列为网络预测的分数, 第二列为实际类别
    '''
    pred, gt = list(), list()

    for X, y in tqdm.tqdm(test_iter):
        if y.shape != torch.Size([]):
            gt += y.tolist()
        else:
            gt.append(y.item())
        X, y = X.to(opt.device), y.to(opt.device)
        net.eval()
        with torch.no_grad():
            y_hat = net(X)
            y_hat = nn.Softmax(dim=1)(y_hat)
            y_hat = torch.argmax(y_hat, dim=1)
            if y.shape != torch.Size([]):
                pred += y_hat.cpu().tolist()
            else:
                pred.append(y_hat.cpu().item())
    
    result = np.array([pred, gt])
    return result

if __name__ == "__main__":
    test_set = getdata.Dataset(opt.test_path, istrain=False)
    test_iter = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

    # 载入网络
    net = get_net(opt.net_name)
    net = net.to(opt.device)
    checkpoint = torch.load(opt.weight_to_load)
    net.load_state_dict(checkpoint['state_dict'])
    
    # 开始推理
    y_pred, y_true = predict(net, test_iter, opt)
    print("acc: ",accuracy_score(y_true, y_pred)) #准确度
    print("pre: ",precision_score(y_true, y_pred, average='binary')) #精确度
    print("recall: ",recall_score(y_true, y_pred, average='binary')) #召回率
    print("f1: ",f1_score(y_true, y_pred, average='binary')) #F1
    confusion = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = confusion.ravel()
    score = TN / (FP + TN)
    print("spec ",score) #spec