import torch
import torch.nn as nn
from torchsummary import summary

class Clinical_Classificator(nn.Module):
    '''
    多模态大模型中的实验室检查数据分类模型, 其中的特征提取层用于多模态融合诊断大模型
    的实验室检查数据的特征提取

    args:
        num_inputs(int): 输入层神经元个数
        num_hidden(int): 隐藏曾神经元个数
        n_classes(int): 类别数
    '''
    def __init__(self, 
                 num_inputs:int, 
                 num_hidden:int,
                 n_classes:int=2) -> None:
        super(Clinical_Classificator, self).__init__()

        self.feature = nn.Sequential(nn.Linear(num_inputs, num_hidden), nn.BatchNorm1d(num_hidden), nn.ReLU())
        self.fc = nn.Linear(num_hidden, n_classes)
    
    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    net = Clinical_Classificator(16, 64, 3)
    print(net)