import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras

from param import MODEL_PATH


class Model:
    """
    模型相关的方法类
    """

    @staticmethod
    def get_model(is_summary=True):
        """
        获取模型，可用于训练或者恢复ckpt参数来使用
        :param is_summary: 是否输出模型网络结果
        :return: 实例化的模型类
        """
        model = NetModel()
        model.compile(optimizer='adam',  # 优化器
                      loss=keras.losses.mean_squared_error,  # 损失函数
                      metrics=['accuracy', Model.dice, tf.metrics.AUC()])
        model.build(input_shape=(None, 20))
        # model.call(keras.Input(shape=(None, 20)))
        if is_summary:
            model.summary()  # 输出模型结构
        return model

    @staticmethod
    def categorical_dicePcrossentroy(Y_pred, Y_gt, weight, lamda=0.5):
        """
        hybrid loss function from dice loss and crossentroy
        loss=Ldice+lamda*Lfocalloss
        :param Y_pred:A tensor resulting from a softmax(-1,z,h,w,numclass)
        :param Y_gt: A tensor of the same shape as `y_pred`
        :param gamma:Difficult sample weight
        :param alpha:Sample category weight,which is shape (C,) where C is the number of classes
        :param lamda:trade-off between dice loss and focal loss,can set 0.1,0.5,1
        :return:diceplusfocalloss
        """
        weight_loss = np.array(weight)
        smooth = 1.e-5
        smooth_tf = tf.constant(smooth, tf.float32)
        Y_pred = tf.cast(Y_pred, tf.float32)
        Y_gt = tf.cast(Y_gt, tf.float32)
        # Compute gen dice coef:
        numerator = Y_gt * Y_pred
        numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
        denominator = Y_gt + Y_pred
        denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
        gen_dice_coef = tf.reduce_sum(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
        loss1 = tf.reduce_mean(weight_loss * gen_dice_coef)
        epsilon = 1.e-5
        # scale preds so that the class probas of each sample sum to 1
        output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keep_dims=True)
        # manual computation of crossentropy
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        loss = -Y_gt * tf.log(output)
        loss = tf.reduce_mean(loss, axis=(1, 2, 3))
        loss = tf.reduce_mean(loss, axis=0)
        loss2 = tf.reduce_mean(weight_loss * loss)
        total_loss = (1 - lamda) * (1 - loss1) + lamda * loss2
        return total_loss

    @staticmethod
    def dice(act, pre):
        """
        dice计算函数
        :param act:
        :param pre:
        :return:
        """
        y_true_f = tf.cast(tf.reshape(act, [-1]), tf.float32)
        y_pred_f = tf.nn.sigmoid(pre)
        y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        dice: float = (2. * intersection) / (union + 0.00001)
        if (tf.reduce_sum(pre) == 0) and (tf.reduce_sum(act) == 0):
            dice = 1.0
        return dice

    @staticmethod
    def get_model_path() -> pathlib.Path:
        """
        检查并获取模型保存路径，目录不存在就创建目录
        :return 模型保存路径Path
        """
        path = pathlib.Path(MODEL_PATH)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def save_model(model: keras.Model):
        """
        保存模型数据
        :param model: 模型
        """
        model_path = Model.get_model_path()
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.save(model_path.joinpath('model'))

    @staticmethod
    def restore_model():
        """
        恢复模型数据，不存在就返回None
        :return: 模型实例或者None
        """
        model_path = Model.get_model_path()
        if not model_path.joinpath('checkpoint').exists():
            return None
        model = Model.get_model(False)
        checkpoint = tf.train.Checkpoint(model=model)
        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        checkpoint.restore(latest_checkpoint).expect_partial()
        return model


class NetModel(keras.Model):
    """
    网络模型
    """

    def __init__(self):
        """
        初始化相关模型
        """
        super().__init__()
        self.net = keras.Sequential([
            keras.layers.Dense(256 * 272, use_bias=False, input_shape=(20,)),  # 密接层，shape=(None, 20)
            keras.layers.BatchNormalization(),  # 规范化层，加快学习收敛，shape=(None, 20)
            keras.layers.LeakyReLU(),  # 激活层，可以给学习率中的所有负值赋予一个非零斜率，shape=(None, 20)
            keras.layers.Reshape((256, 272, 1)),  # 将变量转换为shape形式，shape=(None, 256, 272, 1)
            keras.layers.Conv2DTranspose(1, (4, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'),
            # 逆（转置）卷积操作，shape=(None, 256, 272, 1)
        ])

    def call(self, inputs, training=None, mask=None):
        """
        模型调用函数
        :param inputs: 输入给模型的数据
        :param training: 布尔值或布尔标量张量，指示是否以训练模式或推理模式运行“网络”
        :param mask: 掩码或掩码列表。掩码可以是布尔张量或无（无掩码）。
        :return: 模型执行的输出结果
        """
        out = self.net(inputs)
        # out = self.seq_res(out)
        return out
