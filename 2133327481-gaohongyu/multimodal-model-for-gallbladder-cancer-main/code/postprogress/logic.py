import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

import callback
from data import Data
from model import Model
from param import TRAIN_EPOCH, BATCH_SIZE, CSV_PATH, TRAIN_PROPORTION, VAL_STEP, \
    TRAIN_STEP, VAL_EPOCH, MODE


class Main:
    """
    主操作逻辑类
    """
    @staticmethod
    def menu() -> bool:
        """
        程序主菜单
        :return: 是否需要继续执行
        """
        global MODE
        i = MODE
        MODE = '0'
        if i == '0':
            print('【1】训练')
            print('【2】测试')
            print('【其它】结束程序')
            i = input('操作：')
        match i:
            case '1':
                print('准备训练模型...')
                Train.start()
                return True
            case '2':
                print('准备测试模型...')
                Test.start()
                return True
        return False

class Train:
    @staticmethod
    def start():
        print('初始化模型...')
        model = Model.get_model()
        print('初始化数据...')
        train_ds, val_ds = Data.get_dataset(CSV_PATH, BATCH_SIZE, TRAIN_PROPORTION)
        print('初始化回调...')
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.0001,
            patience=10
        )
        print('开始训练...')
        history = model.fit(
            train_ds,
            steps_per_epoch=TRAIN_STEP,
            validation_data=val_ds,
            validation_steps=VAL_STEP,
            epochs=TRAIN_EPOCH,
            callbacks=[early_stopping, callback.Callback()]
        )
        Train.show_train_result(history)
        Train.eval_model(val_ds, model)
        Model.save_model(model)

    @staticmethod
    def show_train_result(history):
        """
        可视化模型训练相关参数变化
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        dice = history.history['dice']
        val_dice = history.history['val_dice']
        roc = history.history['auc']
        val_roc = history.history['val_auc']
        epochs_range = range(len(acc))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, acc, label='训练Accuracy')
        plt.plot(epochs_range, val_acc, label='验证Accuracy')
        plt.legend(loc='upper right')
        plt.title('Accuracy')
        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='训练Loss')
        plt.plot(epochs_range, val_loss, label='验证Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, dice, label='训练Dice')
        plt.plot(epochs_range, val_dice, label='验证Dice')
        plt.legend(loc='upper right')
        plt.title('Dice')
        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, roc, label='训练Roc')
        plt.plot(epochs_range, val_roc, label='验证Roc')
        plt.legend(loc='upper right')
        plt.title('Roc')
        plt.show()

    @staticmethod
    def eval_model(val_ds, model):
        eva = model.evaluate(val_ds, steps=VAL_EPOCH)
        print('评估结果', eva)


class Test:
    @staticmethod
    def start():
        """
        测试模型逻辑
        """
        model = Model.restore_model()
        if model is None:
            print('模型文件不存在，请先训练模型')
            return
        while True:
            print('【输入数据】获取结果')
            print('【0】结束')
            name = None
            i = input('输入：')
            if i == '0':
                break
            i = i.split(',')
            if len(i) == 21:
                name = i[0]
                i.remove(name)
            if len(i) != 20:
                print(f'输入的数据为{len(i)}位，需要的是20位')
                continue
            # ===========数据预测============
            i = np.asarray(i)
            i = tf.convert_to_tensor(i, dtype=tf.float32)
            i = i[None, :]
            i = model.predict(i)
            # ===========结果处理============
            i = i.squeeze()
            nii_pred = np.where(i <= 0, 0, 1)
            # 如果姓名不为None就获取一个原图
            nii_act = None
            if name is not None:
                nii_act = Data.read_nii_data(Data.get_nii_path(CSV_PATH), name)
                nii_act = Data.convert_to_binary(nii_act)
            # 边缘处理
            # 先获取图片
            fig = plt.figure(frameon=False)
            plt.imshow(i, cmap='gray')
            canvas = fig.canvas
            plt.axis('off')
            # 设置画布大小（单位为英寸），每1英寸有100个像素
            fig.set_size_inches(272 / 100, 256 / 100)
            # plt.gca()表示获取当前子图"Get Current Axes"。
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            buffer = io.BytesIO()  # 获取输入输出流对象
            canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
            data = buffer.getvalue()  # 获取流的值
            buffer.write(data)
            image = np.asarray(Image.open(buffer))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 然后处理该图片，得到填充轮廓二值图
            nii_cnt = Data.fill_image(image)

            # ===========数据展示============
            plt.close()
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            # plt.subplot(2, 3, 1)
            plt.imshow(nii_pred, cmap='gray')
            plt.title('预测图')
            plt.show()
            plt.imshow(nii_cnt, cmap='gray')
            plt.title('填充处理')
            plt.show()
            if nii_act is not None:
                plt.imshow(nii_act, cmap='gray')
                plt.title('原图')
                plt.show()
                # 当有原图时，处理获得差异度>=0.97的二值数据
                nii_handle, nii_final = Data.handle_util_result(nii_cnt, nii_act, 1.00)
                # nii_handle, nii_final = Data.handle_util_result(nii_cnt, nii_act, 0.97)
                plt.imshow(nii_handle, cmap='gray')
                plt.title('综合处理')
                plt.show()
                plt.imshow(nii_final)
                plt.title('最终结果')
                plt.show()
