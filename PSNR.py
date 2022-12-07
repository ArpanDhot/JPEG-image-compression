import cv2
import numpy as np
from math import *
import cv2


def PSNR(o, c):
    original = cv2.imread(o)
    compressed = cv2.imread(c)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def MSE(o,c):
    original = cv2.imread(o)
    compressed = cv2.imread(c)
    mse = np.mean((original - compressed) ** 2)
    return mse