# ================全局参数==================
# 当前程序运行模式，0手动选择，1自动训练，2自动测试，注意是字符串不是数字

MODE = '0'
# 训练迭代次数，有回调函数检查当过拟合发生就停止，所以此数可不太关心
TRAIN_EPOCH = 200
# 训练一次阶段的步数，Step
TRAIN_STEP = 10
# 验证迭代次数
VAL_EPOCH = 10
# 验证步数
VAL_STEP = 5
# 训练中一次喂入网络的数据量，批大小
BATCH_SIZE = 12
# 训练集占比，0~1的小数，余下部分是验证集
TRAIN_PROPORTION = 0.8
# 学习率
LEARN_RATE = 0.001
# csv数据文件路径，nii文件需在同路径的nii文件夹内，如data/nii/xx.nii
CSV_PATH = 'data/data.csv'
# 模型保存路径
MODEL_PATH = 'cache/model'
# nii二值化图存放路径
NII_PATH = 'cache/nii'
# ================全局参数==================