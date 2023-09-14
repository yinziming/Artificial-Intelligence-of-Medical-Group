import numpy as np
import scipy.ndimage
import random
import cv2
import os
from tqdm import tqdm
import SimpleITK as sitk

class preprocess(object):

    def __init__(self,image, window_wide, window_level):
        self.image = image
        self.window_wide = window_wide
        self.window_level = window_level

    # normalize image to 0~1
    def img_normalize(self,normal = False):
        # the upper grey level(x) is calculated via WL + (WW ÷ 2)
        # the lower grey level(y) is calculated via WL - (WW ÷ 2)
        upper_grey = self.window_level + 0.5*self.window_wide
        lower_grey = self.window_level - 0.5*self.window_wide
        new_image = (self.image - lower_grey) / self.window_wide
        new_image[new_image < 0] = 0
        new_image[new_image > 1] = 1
        if not normal:
            new_image = new_image*255
        return new_image

    def random_rotate_img(self, min_angle, max_angle):
        """
        图像旋转
        random rotation an image

        :param img:         image to be rotated
        :param min_angle:   min angle to rotate
        :param max_angle:   max angle to rotate
        :return:            image after random rotated

        """
        if not isinstance(self.image, list):
            img = [self.image]

        angle = random.randint(min_angle, max_angle)
        center = (img[0].shape[0] / 2, img[0].shape[1] / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        res = []
        for img_inst in img:
            img_inst = cv2.warpAffine(img_inst, rot_matrix,dsize=img_inst.shape[:2],
                                      borderMode=cv2.BORDER_CONSTANT)
            res.append(img_inst)
        if len(res) == 0:
            res = res[0]
        return res

    def random_flip_img(self):
        '''
    	图像平移
        random flip image,both on horizontal and vertical
        :param img:                 image to be flipped
        :return:                    image after flipped
        '''
        flip_val = 0
        if not isinstance(self.image, list):
            res = cv2.flip(self.image, flip_val)  # 0 = X axis, 1 = Y axis,  -1 = both
        else:
            res = []
            for img_item in self.image:
                img_flip = cv2.flip(img_item, flip_val)
                res.append(img_flip)
        return res

def normalized(image, ww, wl):
    upper_grey = wl + 0.5 * ww
    lower_grey = wl - 0.5 * ww
    new_image = (image - lower_grey) / ww
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1
    return new_image


def translateit(image, offset, isseg=False):
    order = 0 if isseg == True else 5
    return scipy.ndimage.interpolation.shift(image, (0, int(offset[0]), int(offset[1])), order=order, mode='nearest')

def rotateit(image, theta, isseg=False):
    order = 0 if isseg == True else 5
    return scipy.ndimage.rotate(image, float(theta), axes=(1, 2), reshape=False, order=order, mode='nearest')


if __name__ == '__main__':
    img_path = r'F:\all_users\gaohy\data\ddm\training_sets\train_120\class2'
    save_path = r'F:\all_users\gaohy\data\ddm\training_sets\train_120_enhanced\class2'
    files = os.listdir(img_path)
    for file in tqdm(files):
        img = sitk.ReadImage(os.path.join(img_path, file))
        data = sitk.GetArrayFromImage(img)
        # new_data = normalized(data,300,40)
        theta = float(np.around(np.random.uniform(-10.0, 10.0, size=1), 2))
        new_data = rotateit(data,theta)
        # offset = list(np.random.randint(-5, 5, size=2))
        # new_data = translateit(data,offset)
        new_image = sitk.GetImageFromArray(new_data)
        # write_path = os.path.join(save_path,file[:-4])+'_translateited'+'.nii'
        write_path = os.path.join(save_path, file[:-4]) + '_rotateited' + '.nii'
        # write_path = os.path.join(save_path,file[:-4])+'processed'+'.nii'
        # write_path = os.path.join(save_path,file)
        sitk.WriteImage(new_image,write_path)
    print('finished')


