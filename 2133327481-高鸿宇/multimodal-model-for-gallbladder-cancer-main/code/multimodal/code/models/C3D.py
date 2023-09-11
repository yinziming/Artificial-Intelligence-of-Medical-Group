import torch
import torch.nn as nn
from utils.config import opt

class C3D(nn.Module):
    '''
    多模态融合模型

    args:
        num_classes(int): 最终输出类别个数, 默认为2
        mode(str): 模型生成模式, CL_O: 仅实验室数据(对应实验一)、RA_O:仅放射组学数据(对应实验二)、CT_O: 仅CT数据(对应实验三)、
                                CL_RA: 实验室数据+放射组学数据(对应实验四)、CL_CT: 实验室数据+CT数据(对应实验五)、
                                RA_CT: 放射组学数据+CT数据(对应实验六)、CL_RA_CT: 实验室检查数据+放射组学数据+CT数据(对应实验七)、
                                CL_RA_CT_W:带权重的实验室检查数据+放射组学数据+CT数据(对应实验八)
    '''
    def __init__(self, num_classes=2, mode='CT_O'):
        super(C3D, self).__init__()

        self.mode = mode

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3))#, padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3))#, padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(1536, 512)
        self.fc7 = nn.Linear(512, 24)

        if self.mode == 'CL_O':
            self.fc8 = nn.Linear(6, num_classes)
        elif self.mode == 'RA_O':
            self.fc8 = nn.Linear(14, num_classes)
        elif self.mode == 'CT_O':
            self.fc8 = nn.Linear(24, num_classes)
        elif self.mode == 'CL_RA':
            self.fc8 = nn.Linear(14+6, num_classes)
        elif self.mode == 'CL_CT':
            self.fc8 = nn.Linear(6+24, num_classes)
        elif self.mode == 'RA_CT':
            self.fc8 = nn.Linear(14+24, num_classes)
        elif self.mode == 'CL_RA_CT':
            self.fc8 = nn.Linear(14+6+24, num_classes)
        else:
            self.fc8 = nn.Linear(14+6+24, num_classes)

        self.dropout = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.__init_weight()

        self.weight = torch.tensor([0.3747, 0.2039, 0.4214], device=opt.device)
        self.with_ct = ['CT_O', 'CL_CT', 'RA_CT', 'CL_RA_CT', 'CL_RA_CT_W']

    def forward(self, CT, clinical, ra):
        if self.mode not in self.with_ct:
            if self.mode == 'CL_O':
                x = clinical
            elif self.mode == 'RA_O':
                x = ra
            elif self.mode == 'CL_RA':
                x = torch.cat((clinical,ra),dim=1)
        else:
            x = self.relu(self.conv1(CT))
            x = self.pool1(x)

            x = self.relu(self.conv2(x))
            x = self.pool2(x)

            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)

            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)

            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)

            x = x.view(-1, 1536)
            x = self.sig(self.fc6(x))
            x = self.dropout(x)
            x = self.sig(self.fc7(x))
            x = self.dropout(x)

            if self.mode == 'CT_O':
                x = x
            elif self.mode == 'CL_CT':
                x = torch.cat((x, clinical),dim=1)
            elif self.mode == 'RA_CT':
                x = torch.cat((x, ra),dim=1)
            elif self.mode == 'CL_RA_CT':
                x = torch.cat((x, ra, clinical),dim=1)
            else:
                x = x * self.weight[0]
                ra_w = ra * self.weight[1]
                clinical_w = clinical * self.weight[2]
                x = torch.cat((x, ra_w, clinical_w),dim=1)
        logits = self.fc8(x)
        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()