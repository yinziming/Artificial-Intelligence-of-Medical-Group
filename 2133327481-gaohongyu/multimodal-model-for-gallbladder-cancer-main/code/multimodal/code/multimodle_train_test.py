import os
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, random_split
from models.C3D import C3D
from getdata import Multi_Model_Dataset
from utils.config import opt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc,confusion_matrix,roc_auc_score
from scipy import stats

def train(net, train_iter, opt, loss, optimizer):
    '''
    多模态融合模型训练函数

    args:
        net(nn.Module): 多模态融合模型
        train_iter(DataLoader): 训练数据
        opt(Object): 配置文件中的参数信息
        loss(nn.Module): 损失函数
        optimizer(torch.optim): 优化器
    
    returns:
        net(nn.Module): 已训练完成的多模态融合模型
    '''
    print("training on", opt.device)
    net.to(opt.device)

    for epoch in range(opt.epoch):
        net.train()
        train_loss = 0
        train_num = 0

        for CT, clinical, ra, y in tqdm.tqdm(train_iter, 'train epoch:'+str(epoch)):
            optimizer.zero_grad()
            CT, clinical, ra, y = CT.to(opt.device), clinical.to(opt.device), ra.to(opt.device), y.to(opt.device)
            y_hat = net(CT, clinical, ra)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_num += 1
            train_loss += l
        train_loss /= train_num
        print("epoch:%d loss:%f" %(epoch, train_loss))
        torch.cuda.empty_cache()
    print('finish!')
    return net

def predict(net, test_iter, opt):
    '''
    多模态融合模型预测函数

    args:
        net(nn.Module): 已训练完成的多模态融合模型
        test_iter(DataLoader): 测试数据
    
    returns:
        y_true(ndarray): 测试集数据金标准
        y_hat(ndarray): 网络模型的预测数据
    '''

    y_true, y_hat = list(), list()
    net.to(opt.device)
    net.eval()
    for CT, clinical, ra, y in tqdm.tqdm(test_iter):
        CT, clinical, ra = CT.to(opt.device), clinical.to(opt.device), ra.to(opt.device)
        y_pred = net(CT, clinical, ra)
        y_true += y.numpy().tolist()
        y_hat += y_pred.cpu().detach().numpy().tolist()
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    return y_true, y_hat

def draw_and_save(y_true=None, y_hat=None):
    '''
    数据分析函数, 用于保存数据并绘制图像, 输出分析结果
    
    args:
        y_true(ndarray): 测试集的金标准
        y_hat(ndarray): 多模态数据融合模型的预测数据
    '''
    if y_true is None or y_hat is None:
        # 若 y_true 或 y_hat为空, 说明这两个数据已经保存的本地, 因此需要从本地读取相关数据
        label_path = os.path.join(opt.multi_model_data_save_base_path, opt.experiment, 'y_true.csv')
        pred_path = os.path.join(opt.multi_model_data_save_base_path, opt.experiment, 'y_hat.csv')
        y_true_df = pd.read_csv(label_path)
        y_hat_df = pd.read_csv(pred_path)
        y_true = y_true_df.values[::, 1:]
        y_hat = y_hat_df.values[::, 1:]
    else:
        label_path = os.path.join(opt.multi_model_data_save_base_path, opt.experiment)
        pred_path = os.path.join(opt.multi_model_data_save_base_path, opt.experiment)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
        df = pd.DataFrame(y_true, columns=['label_0', 'label_1'])
        df.to_csv(label_path+'/y_true.csv')
        df = pd.DataFrame(y_hat, columns=['label_0', 'label_1'])
        df.to_csv(pred_path+'/y_hat.csv')
    
    # 绘制ROC
    y_true = y_true[::, 1]
    pred_score = y_hat[::, 1]
    FPR,TPR,threshold=roc_curve(y_true, pred_score, pos_label=1)
    AUC=auc(FPR,TPR)
    ci = confidenceInterval(y_true, pred_score, len(y_true), 'auc', bootstraps=50)
    print(f'auc: {AUC}, CI:{ci}')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(dpi=800)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate',size=10)
    plt.ylabel('True Positive Rate',size=10)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.plot(FPR,TPR,color='darkorange',label='AUC={:.4f}'.format(AUC))
    plt.plot([0, 1], [0, 1], color='navy',lw=3, linestyle='--')
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.legend(loc="lower right",fontsize=10)
    fig_save_path = os.path.join(opt.multi_model_data_save_base_path, opt.experiment)
    if not os.path.exists(fig_save_path):
        os.mkdir(fig_save_path)
    plt.savefig(fig_save_path+"/ROC.jpg", bbox_inches='tight')
    # plt.show()

    # 计算acc, 精确度, 召回率和F1
    y_pred = np.argmax(y_hat, axis=1)
    score_name = ['acc', 'specificity', 'sensitivity', 'precision', 'f1']
    for each_name in score_name:
        score = get_scores(y_true, y_pred, each_name)
        ci = confidenceInterval(y_true, y_pred, len(y_true), each_name, bootstraps=50)
        print(f'{each_name}: {score}, CI:{ci}')
    confusion = confusion_matrix(y_true, y_pred)
    print('confusion matrix:')
    print(confusion)

def confidenceInterval(y, pred, fold_size, CIname, bootstraps=50):
    # 采样100个点
    statistics = []
    df = pd.DataFrame(columns=['y', 'pred'])
    df.loc[:, 'y'] = y
    df.loc[:, 'pred'] = pred
    df_pos = df[df.y == 1]
    df_neg = df[df.y == 0]
    prevalence = len(df_pos) / len(df)
    for i in range(bootstraps):
        pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
        neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

        y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
        pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
        score = get_scores(y_sample, pred_sample, CIname)
        statistics.append(score)
    
    # 计算置信区间
    n = len(statistics)
    ci = stats.t.interval(0.95, n - 1, np.mean(statistics), stats.sem(statistics))
    ci = np.array(ci)
    return ci

def get_scores(y, y_pred, score_name):
    if (score_name == 'auc'):
        score = roc_auc_score(y, y_pred)
    if (score_name == 'acc'):
        score = accuracy_score(y, y_pred)
    if (score_name == 'sensitivity'):
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.ravel()
        score = TP/(TP+FN)
    if (score_name == 'specificity'):
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.ravel()
        score = TN / (FP + TN)
    if(score_name == 'precision'):
        score = precision_score(y, y_pred, zero_division=0)
    if(score_name == 'f1'):
        score = f1_score(y, y_pred, zero_division=0)
    return score

if __name__ == "__main__":
    y_true, y_hat = None, None
    dataset = Multi_Model_Dataset(opt)
    train_set, test_set=random_split(dataset,
                                     [round(0.8*len(dataset)),round(0.2*len(dataset))], # 数据集划分比例
                                     generator=torch.Generator().manual_seed(42))       # 随机种子
    batch_size = 32
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    net = C3D(2, opt.experiment)
    # 设置loss函数
    loss = nn.MSELoss()
    # 设置优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # trained_net = train(net, train_iter, opt, loss, optimizer)
    # y_true, y_hat = predict(trained_net, test_iter, opt)
    
    # 保存数据绘制图像
    draw_and_save(y_true, y_hat)