import pathlib
import random

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage import morphology
from tensorflow.python.data import AUTOTUNE

from param import NII_PATH


# 数据读写和转换
class Data:
    """
    数据处理方法类
    """

    @staticmethod
    def read_csv(csv_path: str) -> dict:
        """
        读取csv文件
        :param csv_path: csv文件路径
        :return: 读取得到的数据，dict格式
        """
        csv = pd.read_csv(csv_path, sep=',', encoding='utf-8')
        csv = csv.dropna(how='all')
        ret = {}
        row, _ = csv.shape  # 行，列
        for i in range(row):
            # 一行的数据，包括第一列名字
            line = csv.iloc[i].tolist()
            name = line[0]
            line.remove(name)
            ret[name] = line
        # 打乱数据，确保每次开始训练都产生不同数据
        random.shuffle(list(ret.keys()))
        return ret

    @staticmethod
    def get_nii_path(csv_path: str) -> pathlib.Path:
        """
        通过csv路径得到csv所在目录，为获取该目录下nii目录中的nii数据做准备
        :param csv_path: csv路径
        :return: nii数据所在目录
        """
        csv_path = pathlib.Path(csv_path)
        return csv_path.parent.joinpath('nii')

    @staticmethod
    def read_nii_data(nii_path: pathlib.Path, nii_name: str) -> np.ndarray:
        """
        读取给定名称的nii数据
        :param nii_path: pathlib.Path格式的nii数据目录
        :param nii_name: 要读取的nii名称，不用传入后缀
        :return: nii数据
        """
        nii = nib.load(nii_path.joinpath(nii_name + '.nii'))
        nii = nii.get_fdata()
        return nii

    @staticmethod
    def convert_to_binary(data: np.ndarray) -> np.ndarray:
        """"
        将三维数组转换为二维数组
        """
        # height 256, width 272, depth n
        # ndarray 坐标轴 z y x
        _, _, depth = data.shape
        # 声明结果数组
        ret: np.ndarray = data[:, :, [0]]
        # data.sum(axis=0)
        # np.where(data == 0, 0, 1)
        for i in range(1, depth):  # 1~depth-1
            ret = ret + data[:, :, [i]]
        # 去掉为1的维度
        ret = ret.squeeze(axis=2)
        # 二值处理
        ret = np.where(ret == 0, 0, 1)
        return ret

    @staticmethod
    def get_one_nii_binary(csv_path, name):
        """
        获取一个二值化的nii数据
        :param csv_path: csv路径
        :param name: 姓名
        :return: 数据
        """
        path = Data.get_nii_path(csv_path)
        data = Data.read_nii_data(path, name)
        data = Data.convert_to_binary(data)
        return data

    @staticmethod
    def get_nii_binary_path() -> pathlib.Path:
        """
        检查并获取处理过后二值图保存路径，目录不存在就创建目录
        :return 二值图保存路径Path
        """
        path = pathlib.Path(NII_PATH)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def save_binary_data(name: str, data: np.ndarray):
        """
        保存二值化图片，注意如果该图片已存在就不会保存
        :param name: 图片名称
        :param data: 二值化数据
        """
        path = Data.get_nii_binary_path()
        path = path.joinpath(f'{name}.jpg')
        if not path.exists():
            plt.axis('off')
            plt.imshow(data, cmap='gray')
            plt.savefig(str(path))
            plt.close()

    @staticmethod
    def preprocess(tags: tf.Tensor, nii: tf.Tensor):
        """
        数据处理函数，将x,y处理成所需要的格式，方便查看数据信息的同时尝试保存处理的nii
        :param tags: 除开姓名外的数据
        :param nii: 二值化数据
        :return: 除开姓名外的数据，二值化数据
        """
        # x = x[None, :]
        # x = x * 100
        # x = tf.convert_to_tensor(x, dtype=tf.int32)
        print('---x.shape---', tags.shape)
        print('---y.shape---', nii.shape)
        print('x', tags)
        print('y', nii)
        return tags, nii

    @staticmethod
    def get_dataset(csv_path: str, batch_size: int, proportion: float = 0.8):
        """
        获取训练数据集
        :param csv_path csv数据文件路径
        :param batch_size 批大小
        :param proportion 训练集占比
        :return: 训练集，测试集
        """
        # =============读取数据===============
        data = Data.read_csv(csv_path)
        names = list(data.keys())
        tags = list(data.values())
        niis = []
        nii_path = Data.get_nii_path(csv_path)
        for name in names:
            nii = Data.read_nii_data(nii_path, name)
            nii = Data.convert_to_binary(nii)
            niis.append(nii)
            # 保存二值化数据到预定目录
            Data.save_binary_data(name, nii)
        # =============划分数据===============
        train_size = int(len(data) * proportion)
        # 前百分之六十用于训练
        train_tags = tags[:train_size]
        train_niis = niis[:train_size]
        # 后百分之四十用于验证
        val_tags = tags[train_size:]
        val_niis = niis[train_size:]
        # =============初始化数据集===============
        train_ds = tf.data.Dataset.from_tensor_slices((train_tags, train_niis))
        val_ds = tf.data.Dataset.from_tensor_slices((val_tags, val_niis))
        train_ds = train_ds.map(Data.preprocess).batch(batch_size).shuffle(100).prefetch(AUTOTUNE).repeat()
        val_ds = val_ds.map(Data.preprocess).batch(batch_size).prefetch(AUTOTUNE).repeat()
        return train_ds, val_ds

    @staticmethod
    def get_differ(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """
        计算差异度，传入相同shape的两个二值化数组
        二值数组求和相当于得到面积，arr3是arr1和arr2的并集
        1减去arr1在arr3中的占比，相当于arr2比arr1多出的比例，越小表示arr1占比越大，这个比例再乘以arr3的面积
        得到arr2比arr1多出的面积，arr2面积减去多出面积，除以arr2面积得到arr1占arr2的比例

        改：更加简便的是arr1占  arr3比例 / arr2占arr3比例
        改2，x/y / z/y = x / z，所以上式不能用于算占比，这里换成直接创建一个2有1没有的区域，算比例
        :param arr1 二值数组
        :param arr2 另一个二值数组
        :return: 差异度，-1~1，如果包含arr2则会返回1，arr1完全被arr2包含就是-1
        """
        # 合并数组，并集
        # arr3 = np.where((arr1 == 0) & (arr2 == 0), 0, 1)
        arr3 = np.where((arr1 == 0) & (arr2 == 1), 1, 0)
        # n1 = np.sum(arr1)
        n2 = np.sum(arr2)
        n3 = np.sum(arr3)

        return 1 - (n3 / n2)

    @staticmethod
    def fill_image(image):
        """
        处理图像获得图像集中轮廓
        :param image: 图片
        :return: 消除小面积后的二值数组
        """
        # 拷贝图片，防止对原图进行操作
        image = image.copy()
        # 小区域消除处理
        _, thresh = cv2.threshold(image, 127, 255, type=cv2.THRESH_BINARY)
        thresh = thresh > 0
        ret = morphology.remove_small_objects(thresh, 60)
        ret = np.where(ret, 1, 0)
        # 横向填充
        for row in ret:
            first = (row != 0).argmax(axis=0)
            last = [i for i, e in enumerate(row) if e != 0]
            if len(last) == 0:
                continue
            last = last[len(last) - 1]
            # 从开始到最后之间的所有区域补1
            for i in range(first, last):
                row[i] = 1
        # 纵向填充
        for col in range(ret.shape[1]):
            col = ret[:, col]
            first = (col != 0).argmax(axis=0)
            last = [i for i, e in enumerate(col) if e != 0]
            if len(last) == 0:
                continue
            last = last[len(last) - 1]
            # 从开始到最后之间的所有区域补1
            for i in range(first, last):
                col[i] = 1
        # 寻找等高线
        # contours, hierarchy = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # ret = np.zeros(image.shape, dtype=np.uint8)
        # epsilon = ret.shape[0] / 256
        # approx = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
        # cv2.polylines(ret, approx, True, (0, 255, 0), 2)  # green
        # cv2.drawContours(ret, contours, -1, (255, 0, 0), 2)  # blue
        # cv2.drawContours(ret, contours, -1, (255, 255, 255), 2)
        # min_side_len = ret.shape[0] / 32  # 多边形边长的最小值 the minimum side length of polygon
        # min_poly_len = ret.shape[0] / 16  # 多边形周长的最小值 the minimum round length of polygon
        # min_side_num = 3  # 多边形边数的最小值
        # min_area = 16.0  # 多边形面积的最小值
        # approx = [cv2.approxPolyDP(cnt, min_side_len, True) for cnt in contours]  # 以最小边长为限制画出多边形
        # approx = [approx for approx in approx if
        #            cv2.arcLength(approx, True) > min_poly_len]  # 筛选出周长大于 min_poly_len 的多边形
        # approx = [approx for approx in approx if len(approx) > min_side_num]  # 筛选出边长数大于 min_side_num 的多边形
        # approx = [approx for approx in approx if cv2.contourArea(approx) > min_area]  # 筛选出面积大于 min_area_num 的多边形
        # cv2.polylines(ret, approx, True, (255, 255, 255), 2)
        return ret

    @staticmethod
    def handle_util_result(arr1: np.ndarray, arr2: np.ndarray, min_differ: float = 0.97):
        """
        不断填充处理直到获得差异度小于给定数值的数组，数组shape为(256,272)
        :param arr1 需要包含另一个数组的数组
        :param arr2 需要被包含的数组
        :param min_differ 最小差异度，当arr1包含时为1，范围0~1
        :return: 满足min_differ的结果
        """
        ret = arr1.copy()
        shape = ret.shape
        last_differ = 0.  # 因为部分数据处理未知原因无法继续扩大，设立此参数
        same_time = 3  # 如果differ相同达到次数就不再处理
        print('处理中...')
        str_format = '\r处理中，当前占比：%.4f'
        while True:
            ret = np.reshape(ret, shape)
            differ = Data.get_differ(ret, arr2)
            if differ == last_differ:
                if same_time <= 0:
                    break
                same_time = same_time - 1
            last_differ = differ
            print(str_format % differ, end='')
            if differ >= min_differ:
                break
            # 占比不达标，扩展arr1
            # 扩展量基础值为1，即四面八方延伸1
            base = 1
            # 工作量太大，只实现1
            # # 占比小于0.8时就为2，加快一点
            # if differ < 0.8:
            #     base = 2
            # # 占比小于0.4时就为3，加快两点
            # if differ < 0.8:
            #     base = 2
            # 所以得到一个切片的size就是base*base大小
            s = base * 2
            for i in range(shape[0] - s):
                for j in range(shape[1] - s):
                    # i, j就是切片开始位置，于是得到base长宽的正方形
                    b = ret[i:i + s, j:j + s]  # 2*2的块
                    # 判断是否有数据，没有就直接继续，理论上每次只走一步，出现数据只会在右或者下出现一格或者一排或者一列
                    # 即和的最大值不会超过base
                    # 扩展的数据为2，目标为1，后续再二值化
                    if b[1][1] == 1:
                        b[1][0] = 2 if b[1][0] == 0 else b[1][0]
                        b[0][0] = 2 if b[1][0] == 0 else b[0][0]
                        b[0][1] = 2 if b[0][1] == 0 else b[0][1]
                    if b[0][0] == 1:
                        b[1][0] = 2 if b[1][0] == 0 else b[1][0]
                        b[1][1] = 2 if b[1][0] == 0 else b[1][1]
                        b[0][1] = 2 if b[0][1] == 0 else b[0][1]
                    if b[0][1] == 1:
                        b[1][0] = 2 if b[1][0] == 0 else b[1][0]
                        b[1][1] = 2 if b[1][0] == 0 else b[1][1]
                        b[0][0] = 2 if b[0][0] == 0 else b[0][0]
                    if b[1][0] == 1:
                        b[0][1] = 2 if b[0][1] == 0 else b[0][1]
                        b[1][1] = 2 if b[1][0] == 0 else b[1][1]
                        b[0][0] = 2 if b[0][0] == 0 else b[0][0]
                    ret[i:i + s, j:j + s] = b
            # 二值化
            ret = np.where(ret == 0, 0, 1)

        ret = np.reshape(ret, shape)
        print('\n填补处理完成')
        # ==============原图轮廓与结果合并=============
        ret2 = cv2.cvtColor(np.uint8((ret.copy() * 255)[:,:,np.newaxis]), cv2.COLOR_GRAY2RGB)
        res = cv2.cvtColor(np.uint8(arr2.copy()[:,:,np.newaxis]), cv2.COLOR_GRAY2RGB)
        res = cv2.cvtColor(np.uint8(res), cv2.COLOR_RGB2GRAY)
        contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        epsilon = ret2.shape[0] / 256
        approx = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
        # 将目标轮廓写入结果
        cv2.polylines(ret2, approx, True, (255, 0, 0), 2)
        print('合并处理完成')
        return ret, ret2
