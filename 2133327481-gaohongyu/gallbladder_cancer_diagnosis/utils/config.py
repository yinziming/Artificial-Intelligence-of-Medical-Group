import numpy as np

class Config:
    # 实验室检查数据分类模型训练参数
    # 数据集路径
    data_set_path = 'data/gallbladder_detection/dataset/classification/ClinicalData/dataset.xlsx'
    train_set_path = 'data/gallbladder_detection/dataset/classification/ClinicalData/train_set.xlsx'
    valid_set_path = 'data/gallbladder_detection/dataset/classification/ClinicalData/valid_set.xlsx'
    test_set_path = 'data/gallbladder_detection/dataset/classification/testset/test_set.xlsx'

    # 归一化参数, 用于处理实验室检查数据
    max_num = np.array([1., 1., 88, 21.4, 1.95, 47, 7.91, 22.9, 21.85, 170, 561,
                        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    
    min_num = np.array([ 0.  ,  0.  , 22.  ,  8.9 ,  0.78, 21.1 ,  1.  , 11.4 ,  1.22,
                         60.  ,  5.8 ,  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # 输入特征数
    num_inputs = 22

    # 隐藏层个数
    num_hidden = 512

    # batch size
    clinical_batch_size = 4

    # 学习率
    clinical_lr = 1e-3

    # 训练轮数
    clinical_epoch:int = 100

    # 已保存的模型
    clinical_weight_to_load:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/clinical/clinical_model.pth.tar'

    # 权重保存路径
    clinical_weight_dir:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/clinical'

    # log信息保存路径
    clinical_log_dir:str = 'workspace/graduation_project/logs/gallbladder_cancer_diagnosis/clinical'

    # clinical模型预测结果保存路径
    clinical_data_saved_base_path:str = 'workspace/graduation_project/predict/gallbladder_cancer_diagnosis/clinical'

    # moco相关配置参数
    # moco输出特征维度
    dim:int = 128

    # 动量参数
    m:float = 0.999

    # 负样本队列长度
    K:int = 32768

    # moco 温度系数
    T:float = 0.07

    # moco batch_size 常规为128
    moco_batch_size:int = 128

    # moco训练轮数
    moco_epoch:int = 800

    # moco训练学习率
    moco_lr:float = 0.015

    # moco权重保存路径
    moco_weight_dir:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/CT/single_slice/moco'

    # 已保存的moco预训练权重
    moco_saved_weight_path:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/CT/single_slice/moco/resnet_18/moco_v2resnet_18.pth.tar'

    # moco log信息保存路径
    moco_log_dir:str = 'workspace/graduation_project/logs/gallbladder_cancer_diagnosis/moco'

    # 单ct诊断参数
    # 数据集路径
    single_ct_path = 'data/gallbladder_detection/dataset/classification/CT/single_slice'

    # 测试集路径
    single_ct_testset_path:str = 'data/gallbladder_detection/dataset/classification/testset/sliceforgt'

    # 模型batch_size
    single_ct_batch_size:int = 128

    # 训练轮数
    single_ct_epoch:int = 300

    # 学习率
    single_ct_lr:float = 5e-3

    # 是否使用预训练权重
    pretrained:bool = True

    # 权重保存路径
    single_ct_weight_dir:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/CT/single_slice'

    # log信息保存路径
    single_ct_log_dir:str = 'workspace/graduation_project/logs/gallbladder_cancer_diagnosis/single_slice'

    # 已保存的单CT诊断模型权重
    single_ct_saved_weights_path:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/CT/single_slice/resnext_50_32x4d_moco/single_ct_model.pth.tar'

    # 单ct模型预测结果保存路径
    single_ct_data_saved_base_path:str = 'workspace/graduation_project/predict/gallbladder_cancer_diagnosis/single_slice'

    # 多ct诊断模型参数
    # 训练集路径
    multi_ct_path = 'data/gallbladder_detection/dataset/classification/CT/multi_slice'

    # 测试集路径
    multi_ct_testset_path:str = 'data/gallbladder_detection/dataset/classification/testset/multi_slice'

    # 取的slice张数
    # 3, 5, 7, 9, 11, 3axis
    n_slice:int = 5

    # 多ct融合方式
    # mean: 求平均值， transformer: 多头注意力后求平均值(使用transformer块的结构)
    # rnn: 使用lstm提取特征
    fusion_mode:str = 'mean'

    # 模型batch_size
    multi_ct_batch_size:int = 16

    # 训练轮数
    multi_ct_epoch:int = 300

    # 学习率
    multi_ct_lr:float = 1e-5

    # 权重保存路径
    multi_ct_weight_dir:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/CT/multi_slice'

    # log信息保存路径
    multi_ct_log_dir:str = 'workspace/graduation_project/logs/gallbladder_cancer_diagnosis/multi_slice'

    # 已保存的多CT诊断模型权重
    multi_ct_saved_weights_path:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/CT/multi_slice/5slice/fusion_mode_mean/resnext_50_32x4d/multi_ct_model.pth.tar'

    # 多ct模型预测结果保存路径
    multi_ct_data_saved_base_path:str = 'workspace/graduation_project/predict/gallbladder_cancer_diagnosis/multi_slice'

    # 三个模型的公共参数
    # 类别个数
    n_class = 2

    # weight_decay
    weight_decay:float = 1e-4

    # device
    device:str = 'cuda:0'

    # 诊断模型选择
    # resnet_18, resnet_34, resnet_50, resnet_101, resnet_152,
    # resnext_50_32x4d, resnext_101_64x4d, resnext_101_32x8d
    # densenet_121, densenet_161, densenet_169, densenet_201
    # convnext_T, convnext_S, convnext_B, convnext_L
    backbone_name:str = 'resnext_50_32x4d'

    # 训练集路径
    KA_ct_path = 'data/gallbladder_detection/dataset/classification/CT'
    KA_dataset_path = 'data/gallbladder_detection/dataset/classification/multi_model'

    # 测试集路径
    KA_ct_testset_path:str = 'data/gallbladder_detection/dataset/classification/testset'
    KA_dataset_testset_path = 'data/gallbladder_detection/dataset/classification/testset'

    # 多模态融合模型隐藏层维度
    ka_hidden_dim:int = 768

    # 多模态融合方式
    # normal knowledge_aware
    KA_fusion_mode:str = 'knowledge_aware'

    # 是否使用多ct联合诊断模型作为ct的encoder
    use_multi_slice_encoder:bool = True

    # 模型batch_size
    KA_batch_size:int = 256

    # 训练轮数
    KA_epoch:int = 10

    # 学习率
    KA_lr:float = 5e-3

    # 权重保存路径
    KA_weight_dir:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/multi_model'

    # log信息保存路径
    KA_log_dir:str = 'workspace/graduation_project/logs/gallbladder_cancer_diagnosis/multi_model'

    # 已保存的多模态融合诊断模型权重
    KA_saved_weights_path:str = 'workspace/graduation_project/weights/gallbladder_cancer_diagnosis/multi_model/ct_encoder_type_multi_slice/fusion_mode_knowledge_aware/resnext_50_32x4d/history/KA_model.pth.tar'

    # 多模态融合模型预测结果保存路径
    KA_data_saved_base_path:str = 'workspace/graduation_project/predict/gallbladder_cancer_diagnosis/multi_model'

opt = Config()