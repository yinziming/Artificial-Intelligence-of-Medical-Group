import os
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from utils.config import opt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc,confusion_matrix,roc_auc_score
from scipy import stats
from get_data import Single_ct_Dataset
from torch.utils.data import DataLoader
from models.get_net import get_net

def predict(net, test_iter, opt):
    '''
    单ct诊断模型预测函数

    args:
        net(nn.Module): 已训练完成的多模态融合模型
        test_iter(DataLoader): 测试数据
    
    returns:
        y_true(ndarray): 测试集数据金标准
        y_hat(ndarray): 网络模型的预测数据
    '''
    y_true, y_hat = list(), list()
    net.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(test_iter):
            x = x.to(opt.device)
            y_pred = net(x)
            y_pred = nn.Softmax(dim=1)(y_pred)
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
        label_path = os.path.join(opt.single_ct_data_saved_base_path, 'backbone', opt.backbone_name, 'y_true.csv')
        pred_path = os.path.join(opt.single_ct_data_saved_base_path, 'backbone', opt.backbone_name, 'y_hat.csv')
        y_true_df = pd.read_csv(label_path)
        y_hat_df = pd.read_csv(pred_path)
        y_true = y_true_df.values[::, 1]
        y_hat = y_hat_df.values[::, 1]
    else:
        label_path = os.path.join(opt.single_ct_data_saved_base_path, 'backbone', opt.backbone_name)
        pred_path = os.path.join(opt.single_ct_data_saved_base_path, 'backbone', opt.backbone_name)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
        df = pd.DataFrame(y_true, columns=['label'])
        df.to_csv(label_path+'/y_true.csv')
        df = pd.DataFrame(y_hat, columns=['label_0', 'label_1'])
        df.to_csv(pred_path+'/y_hat.csv')
    
    # 绘制ROC
    pred_score = y_hat[::, 1]
    FPR,TPR,threshold=roc_curve(y_true, pred_score, pos_label=1)
    AUC=auc(FPR,TPR)
    ci = confidenceInterval(y_true, pred_score, len(y_true), 'auc', bootstraps=50)
    print(f'auc: {AUC}, CI:{ci}')
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus']=False
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
    fig_save_path = os.path.join(opt.single_ct_data_saved_base_path, 'backbone', opt.backbone_name)
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
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

        y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values]).astype(np.int64)
        pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values]).astype(np.int64)
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
    test_set = Single_ct_Dataset(opt.single_ct_testset_path, 'test_set', False)
    test_iter = DataLoader(test_set, batch_size=opt.single_ct_batch_size, shuffle=False)
    net = get_net(opt.backbone_name, n_classes = opt.n_class)
    weights = torch.load(opt.single_ct_saved_weights_path)['state dict']
    net.to(opt.device)
    net.load_state_dict(weights)
    y_true, y_hat = predict(net, test_iter, opt)
    # # 保存数据绘制图像
    draw_and_save(y_true, y_hat)

    y_hat = np.argmax(y_hat, axis=1)

    acc = (y_hat == y_true).sum() / len(y_hat)
    print(f'acc:{acc}')

    confusion = confusion_matrix(y_true, y_hat)
    print('confusion matrix:')
    print(confusion)