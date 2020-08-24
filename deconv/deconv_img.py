import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import sqrt, tan, log, pi, exp
from scipy.ndimage import gaussian_filter1d
from scipy.stats import multivariate_normal


class Deconv:
    def __init__(self):
        # self.img = cv2.imread('Deconvolution/IMG/0824_Deconv_kernel-5.png', 0)
        self.img = cv2.imread('Deconvolution/IMG/0824_ori_img.png', 0)
        # self.var = 3
        self.eval_sharpness(self.img)
        # self.gauss_blur()
        # self.eval_sharpness(self.img)
        # self.vis_img()

    def gauss_blur(self):
        # for i in range(self.img.shape[1]):
        #     self.img[:,i] = gaussian_filter1d(self.img[:,i], 9)
        row, col = self.img.shape[0], self.img.shape[1]

        A = np.zeros((row, col))
        for i in range(len(A)-self.var):
            width = self.var/2
            x = np.arange(-width, width+1, 1)
            gaussian = multivariate_normal(mean=0, cov=width)
            y = gaussian.pdf(x)
            y = y/sum(y)
            start = int((self.var-1)/2 + i - width)
            end = int((self.var-1)/2 + i + width + 1)
            A[i, start:end] = y
        A_inv = np.linalg.pinv(A)
        self.A, self.A_inv = A, A_inv
        self.img = np.dot(self.A, self.img)

    def vis_img(self):
        img = cv2.normalize(src=self.img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imshow('Example', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def deconv_pinv(self):
        result = np.dot(self.A_inv, self.img)
        self.eval_sharpness(result)
        result = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        cv2.imshow('Example', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def eval_sharpness(self, img, alg='brenner'):
        if alg == 'brenner':
            brenner = 0
            for j in range(len(img)):
                I = img[:, j]
                for i in range(len(I)-2):
                    brenner = brenner + (I[i+2]-I[i])**2
                brenner = brenner * 1. / len(I)
            brenner = brenner * 1. / len(img)
            print('Brenner sharpness = %e' % brenner)
        return

    def deconv_loss(self):
        s_len = 512
        I_len = s_len - self.var + 1

        D = np.r_[1, -2, -1, np.zeros()]
        return


test = Deconv()
print("Orz")
