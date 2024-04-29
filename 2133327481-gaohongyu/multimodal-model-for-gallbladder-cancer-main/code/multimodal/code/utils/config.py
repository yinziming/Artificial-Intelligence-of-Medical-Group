class Opt:
    # 数据集路径
    train_path = 'data/CT/train_set.csv'
    valid_path = 'data/CT/valid_set.csv'
    test_path = 'data/CT/test_set.csv'

    # 所训练的网络 ['Convnext_T', 'Convnext_S', 'Convnext_B', 'Mobilenet', 'InceptionV3', 
    #              'ViT-B-16', 'ViT-B-32']
    net_name = 'Convnext_T'

    # 图像大小 InceptionV3为299*299*299, 其余为224*224*224
    # img_size = (299, 299, 299)
    img_size = (224, 224, 224)

    # 优化器参数
    weight_decay = 1e-3
    lr = 1e-3
    
    # 训练参数
    epoch = 300
    batch_size = 2 # Convnext_B为1， 其余为2
    
    # 已训练的轮数
    trained_epoch = 0

    # 已保存的权重, 用于模型推理或者断点续训练
    weight_to_load:str = 'weights\Convnext_T\Convnext_T_epoch-60_loss-0.693147_acc-0.500000.pth.tar'

    # 是否根据之前的训练权重继续训练
    load_weight = False
    
    # log信息保存路径
    log_dir:str = 'event/'
    
    # weight保存路径
    weight_dir:str = 'weights/'
    
    #device
    device:str = 'cuda:0'

    # experiment(str): 多模态融合实验, CL_O: 仅实验室数据(对应实验一)、RA_O:仅放射组学数据(对应实验二)、CT_O: 仅CT数据(对应实验三)、
    #                                 CL_RA: 实验室数据+放射组学数据(对应实验四)、CL_CT: 实验室数据+CT数据(对应实验五)、
    #                                 RA_CT: 放射组学数据+CT数据(对应实验六)、CL_RA_CT: 实验室检查数据+放射组学数据+CT数据(对应实验七)、
    #                                 CL_RA_CT_W:带权重的实验室检查数据+放射组学数据+CT数据(对应实验八)
    experiment:str = 'CL_RA_CT_W'

    # 多模态融合数据基路径
    multi_model_base_path:str = 'data/'

    # 多模态融合数据结果保存基路径
    multi_model_data_save_base_path:str = 'predict/multimodel/'

opt = Opt()