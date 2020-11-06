import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pylab import *


def noise(img,snr):
    h=img.shape[0]
    w=img.shape[1]
    img1=img.copy()
    sp=h*w   # 计算图像像素点个数
    NP=int(sp*(1-snr))   # 计算图像椒盐噪声点个数
    for i in range (NP):
        randx=np.random.randint(1,h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy=np.random.randint(1,w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random()<=0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx,randy]=0
        else:
            img1[randx,randy]=255
    return img1


def MeanFilter (Imge,dim):      # Image为待处理图像，dim为滤波器的大小dim*dim
    im= array (Imge)
    sigema=0
    for i in range(int(dim/2), im.shape[0] - int(dim/2)):
        for j in range(int(dim/2), im.shape[1] - int(dim/2)):
            for a in range(-int(dim/2), -int(dim/2)+dim):
                for b in range(-int(dim/2), -int(dim/2)+dim):
                    sigema = sigema + img[i + a, j + b]
            im[i, j] = sigema / (dim*dim)
            sigema = 0
    return im


def MedianFilter (Imge, dim):     # Image为待处理图像，dim为滤波器的大小dim*dim
    im=array(Imge)
    sigema=[]
    for i in range(int(dim/2), im.shape[0] - int(dim/2)):
        for j in range(int(dim/2), im.shape[1] - int(dim/2)):
            for a in range(-int(dim/2), -int(dim/2)+dim):
                for b in range(-int(dim/2), -int(dim/2)+dim):
                    sigema.append(img[i + a, j + b])
            sigema.sort()
            im[i, j] = sigema[int(dim*dim/2)]
            sigema = []
    return im


img = cv.imread("E:/cvimages/logo.jpg", 0)
# img1 = cv.imread("E:/cvimages/logo.jpg")
SNR=0.97   # 将椒盐噪声信噪比设定为0.9
noise_img = noise(img,SNR)
img_median = MedianFilter(noise_img, 3)
img_mean = MeanFilter(noise_img, 3)
# img_median = cv.medianBlur(img1,3)  # 中值滤波
# img_mean = cv.blur(img1, (3,3))     # 均值滤波
cv.imshow("img", img)
cv.imshow("noise_img", noise_img)
cv.imshow("imd_median", img_median)
cv.imshow("img_mean", img_mean)
cv.waitKey(0)
cv.destroyAllWindows()


