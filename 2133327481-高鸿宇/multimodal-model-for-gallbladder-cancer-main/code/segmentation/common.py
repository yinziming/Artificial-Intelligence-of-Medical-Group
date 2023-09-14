import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch

MIN_BOUND = -100.
MAX_BOUND = 200.

def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def poly_lr_scheduler(optimizer, base_lr, n_iter, lr_decay_iter=1, max_iter=100, power=0.9):
    if n_iter % lr_decay_iter == 0 and n_iter <= max_iter:
        lr = base_lr * (1 - n_iter / max_iter) ** power
        for param_gourp in optimizer.param_groups:
            param_gourp['lr'] = lr