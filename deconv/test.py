from auvlib.data_tools import std_data, xtf_data
import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import sqrt, tan, sin, log, pi, exp, ceil
from scipy.ndimage import gaussian_filter1d
from scipy.stats import multivariate_normal
from tqdm import tqdm
import copy


class Deconv(object):
    def __init__(self):
        # read xtf data
        # attitude_data = std_data.attitude_entry.read_data("/home/chs/Desktop/Sonar/Deconvolution/data/mbes_attitude.cereal")
        # self.xtf_ping = xtf_data.xtf_sss_ping.parse_file("/home/chs/Desktop/Sonar/Deconvolution/data/SSH-0047-l08s01-20190618-205611.XTF")
        # self.xtf_ping = xtf_data.match_attitudes(self.xtf_ping, attitude_data)[350:-350]

        xtf_file = "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_19.cereal"  # sidescan data 42. 26 21 29 19 18
        self.xtf_ping = xtf_data.xtf_sss_ping.read_data(xtf_file)[200:300]
        # normalize waterfall image
        self.waterfall = xtf_data.make_waterfall_image(self.xtf_ping)
        # self.waterfall = self.normalise_sss_img(self.waterfall)

        # compute scan parameters
        self.speed_lis, self.time_lis, self.slant_range_lis = self.scan_info(self.xtf_ping)

        self.port_len = len(self.xtf_ping[0].port.pings)
        self.stbd_len = len(self.xtf_ping[0].stbd.pings)

    def normalise_sss_img(self, waterfall_image, clip_max=5):  # TODO: try cotan correction
        """Given a sss_waterfall_image from draping, process the
        image and return a normalised version where the column-wise
        mean is set to 1."""

        img_mean = waterfall_image.mean(axis=0)

        # set points around nadir to 0
        waterfall_image = waterfall_image.copy()
        waterfall_image[:, img_mean < 1e-1] = 0

        img_normalised = np.divide(
            waterfall_image,
            img_mean,
            out=np.zeros_like(waterfall_image),
            where=(img_mean != 0))
        img_normalised = np.clip(a=img_normalised, a_min=img_normalised.min(), a_max=clip_max)

        return img_normalised

    def scan_info(self, xtf_ping):
        '''[Average speed of the auv and save scan parameters]
        '''
        # Filter auv speed
        pos = np.array([0, 0])
        time = 0
        speed_lis = []
        time_lis = []
        slant_range_lis = []
        for ping in xtf_ping:
            pos_dif = pos-ping.pos_[:2]
            time_dif = ping.time_stamp_-time
            pos = ping.pos_[:2]
            time = ping.time_stamp_
            speed_lis.append(sqrt(pos_dif[0]**2 + pos_dif[1]**2)/time_dif*1000)
            time_lis.append(time_dif)
            slant_range_lis.append(ping.port.slant_range)

        speed_lis = np.array(speed_lis[1:])
        speed_lis = gaussian_filter1d(speed_lis, 10)  # smooth the speed
        speed_lis = np.r_[speed_lis[0], speed_lis]

        # plt.title('AUV speed')
        # plt.ylim(0,3)
        # plt.scatter(np.arange(len(speed_lis)), speed_lis, s=2)
        # plt.show()

        return speed_lis, time_lis[1:], slant_range_lis

    def eval_sharpness(self, I_conv):
        # TODO: FFT clearence evaluate
        # TODO: evaluate the whole image
        brenner = 0
        for i in range(len(I_conv)-2):
            brenner += (I_conv[i+2]-I_conv[i])**2
        brenner = brenner / len(I_conv)
        print('Brenner sharpness = %e' % brenner)

    def eval_img_sharpness(self, img,  alg='brenner'):
        # img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # img = cv2.equalizeHist(img)
        if alg == 'brenner':
            brenner = 0
            for j in range(len(img)):
                I = img[j]
                for i in range(len(I)-1):
                    brenner = brenner + (I[i+1]-I[i])**2
                brenner = brenner * 1. / len(I)
            brenner = brenner * 1. / len(img)
            # print('Brenner sharpness = %e' % brenner)
            return brenner

    def gauss_para(self, j, beam_angle, side):
        '''[Compute the parameters of deconv gaussion kernel]

        :param speed_lis: [description]
        :type speed_lis: [type]
        :param time_lis: [description]
        :type time_lis: [type]
        :param slant_range_lis: [description]
        :type slant_range_lis: [type]
        :return: [description]
        :rtype: [type]
        '''
        delta_t = np.mean(self.time_lis)/1000  # get average delta t(160ms or 170ms)
        delta_s = self.speed_lis*delta_t  # ping interval in meters
        slant_range_lis = np.array(self.slant_range_lis)

        ping_len = 9956
        r = slant_range_lis * j / ping_len

        # beam_angle = 1.7 * pi / 180  # test parameters
        # beam_angle = 3000./400/40 * pi / 180  # parameters from sonar manul
        w = 2*tan(beam_angle/2)*r  # half power beamwidth
        n = w/delta_s  # beamwidth in pixel
        return n

    def vis_waterfall_img(self, waterfall, title, mode=cv2.WINDOW_FREERATIO):
        waterfall = cv2.normalize(src=waterfall, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        waterfall = cv2.equalizeHist(waterfall)
        cv2.namedWindow(title, mode)  # cv2.WINDOW_FREERATIO/ cv2.WINDOW_KEEPRATIO
        cv2.imshow(title, waterfall)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # xtf_data.show_waterfall_image(test.xtf_ping) # built in vis method

    def make_waterfall_image2(self):
        port_len = len(self.xtf_ping[0].port.pings)
        stbd_len = len(self.xtf_ping[0].stbd.pings)
        row_len = port_len + stbd_len
        img = []
        for ping in self.xtf_ping:
            row = np.zeros(row_len)
            row[stbd_len:] = ping.stbd.pings
            row = row[::-1]
            #row[port_len:] = ping.port.pings
            img.append(row)

        img = np.array(img)
        self.vis_waterfall_img(img, 'port')

    def vis_waterfall_col(self, col):
        plt.figure(0)
        x = np.arange(len(col))
        plt.plot(x, col)
        plt.show()

    def kernel(self, kernel_len, beamwidth, mode):
        x = np.arange(0, kernel_len, 1)
        x = x - np.mean(x)
        if mode == 'gauss':
            var = (0.5*beamwidth)**2/(-2*log(0.5))
            y = np.exp(-x**2 / (2*var)) / (sqrt(var*2*pi))
        if mode == 'gauss2':
            var = (0.5*beamwidth)**2/(-2*log(0.5))
            y = np.exp(-x**2 / (2*var)) / (sqrt(var*2*pi))**2
        if mode == 'flat':
            y = np.ones(kernel_len)
        if mode == 'sin':
            a = pi/(3*beamwidth)
            y = np.cos(a*x)
            y[y < 0] = 0
        if mode == 'sin2':
            a = pi/(3*beamwidth)
            y = np.cos(a*x)
            y[y < 0] = 0
            y = y**2
        return (y/sum(y))

    def deconv_pinv_col(self, j, kernel_len, kernel_mood, side):
        '''[Deconv with pseudo inverse]

        :param j: [description]
        :type j: [type]
        '''

        beamwidth = self.gauss_para(j, side)  # get beamwidth in pixel

        # Take the column j
        if side == 'port':
            I_conv = self.waterfall[:, self.port_len-j-1]
        elif side == 'stbd':
            I_conv = self.waterfall[:, self.stbd_len+j]

        kernel_half = int((kernel_len-1)/2)
        add_up = np.mean(I_conv) * np.ones(kernel_half)
        # Check if var too small
        if np.mean(beamwidth) == 0:
            return np.r_[add_up, I_conv, add_up]

        y = self.kernel(kernel_len, np.mean(beamwidth), kernel_mood)
        if y[kernel_half] == 1:
            return np.r_[add_up, I_conv, add_up]

        # Make convolution matrix A
        row = len(I_conv)
        col = int(len(I_conv) + kernel_len - 1)
        A = np.zeros((row, col))
        for i in range(len(A)):
            y = self.kernel(kernel_len, beamwidth[i], kernel_mood)
            A[i, i:i+kernel_len] = y

        A_inv = np.linalg.pinv(A)
        s = np.dot(A_inv, I_conv)
        return s

    def deconv_pinv(self, kernel_len=5, kernel_mood='sin'):
        start, end = 9000, 9500
        col = end - start
        ori_row = len(self.waterfall)
        test_row = ori_row + kernel_len - 1
        kernel_half = int((kernel_len-1)/2)
        ori_img, test_img = np.zeros((ori_row, col)), np.zeros((test_row, col))

        for j in tqdm(range(start, end)):
            test_img[:, j-start] = self.deconv_pinv_col(j, kernel_len, kernel_mood, 'stbd')
            ori_img[:, j-start] = self.waterfall[:, self.port_len+j]
        test_img = test_img[kernel_half:test_row-kernel_half]
        ori_sharp = self.eval_img_sharpness(ori_img)
        deconv_sharp = self.eval_img_sharpness(test_img)
        improve = (deconv_sharp - ori_sharp)/ori_sharp * 100
        print("ori_img sharpness = %e" % ori_sharp)
        print("deconv sharpenss = %e" % deconv_sharp)
        print("improvment = %f %%" % improve)
        self.vis_waterfall_img(ori_img, 'pinv_result')

    def deconv_RL_col(self, j, kernel_len, kernel_mood, iteration, beam_angle, side):

        beamwidth = self.gauss_para(j, beam_angle, side)

        kernel_half = int((kernel_len-1)/2)

        # Take the column j
        if side == 'port':
            I_conv = self.waterfall[:, self.port_len-j-1]
        elif side == 'stbd':
            I_conv = self.waterfall[:, self.stbd_len+j]

        # Check if var too small
        add_up = np.mean(I_conv) * np.ones(kernel_half)
        if np.mean(beamwidth) == 0:
            return np.r_[add_up, I_conv, add_up]

        y = self.kernel(kernel_len, np.mean(beamwidth), kernel_mood)
        if y[kernel_half] == 1:
            print("beamwidth too small")
            return np.r_[add_up, I_conv, add_up]

        s = np.r_[add_up, I_conv, add_up]  # get the init s

        # Make conv matrix A
        row = len(I_conv)
        col = int(len(I_conv) + kernel_len - 1)
        A = np.zeros((row, col))
        for i in range(len(A)):
            y = self.kernel(kernel_len, beamwidth[i], kernel_mood)
            A[i, i:i+kernel_len] = y

        for k in range(iteration):
            C = np.dot(A, s)
            cache = I_conv / C
            add_up = np.mean(cache) * np.ones(kernel_half)
            cache = np.r_[add_up, cache, add_up]
            cache = np.dot(A, cache)
            add_up = np.mean(cache) * np.ones(kernel_half)
            cache = np.r_[add_up, cache, add_up]
            s = s * cache

        return s

    def deconv_RL(self, kernel_len=5, kernel_mood='sin2', iteration=5, beam_angle=3e-3):
        start, end = self.port_len-4000, self.port_len-2500
        col = end - start
        ori_row = len(self.waterfall)
        test_row = len(self.waterfall) + kernel_len - 1
        kernel_half = int((kernel_len-1)/2)
        ori_img, test_img = np.zeros((ori_row, col)), np.zeros((test_row, col))

        for j in tqdm(range(start, end)):
            ori_img[:, j-start] = self.waterfall[:, self.port_len-j-1]
            test_img[:, j-start] = self.deconv_RL_col(j, kernel_len, kernel_mood, iteration, beam_angle, 'port')
        test_img = test_img[kernel_half:test_row-kernel_half]
        ori_sharp = self.eval_img_sharpness(ori_img)
        deconv_sharp = self.eval_img_sharpness(test_img)
        improve = (deconv_sharp - ori_sharp)/ori_sharp * 100
        print("kernel:" + kernel_mood + ' iter:%d beam_angle:%f' % (iteration, beam_angle))
        print("ori_img sharpness = %e" % ori_sharp)
        print("deconv sharpenss = %e" % deconv_sharp)
        print("improvment = %f %%" % improve)
        self.vis_waterfall_img(test_img, 'pinv_result')
        # self.vis_waterfall_img(ori_img, 'ori_img')
        # return improve, [beam_angle, kernel_mood, kernel_len, iteration]

    def para_search(self):
        beam_angle = [32e-4, 35e-4, 37e-4, 40e-4, 42e-4, 44e-4, 46e-4]  
        kernel_lis = ['gauss', 'gauss2', 'sin', 'sin2']
        kerenl_len_lis = [3, 5, 7]
        iter_lis = [3, 5, 7, 10]
        imp = 0
        for angle in beam_angle:
            for kenel in kernel_lis:
                for kerenl_len in kerenl_len_lis:
                    for iter_num in iter_lis:
                        temp, para = self.deconv_RL(beam_angle=angle, kernel_len=kerenl_len, kernel_mood=kenel, iteration=iter_num)
                        if temp > imp:
                            imp = temp
                            best_para = para
        print(imp, best_para)


test = Deconv()
# test.para_search()
# (264.01465714935625, [0.0046, 'sin2', 7, 10])
test.deconv_RL(beam_angle=46e-4, kernel_len=7, iteration=10)
# test.vis_waterfall_col(test.waterfall[:, 3400])
# test.vis_waterfall_img(test.waterfall, 'test')
# test.make_waterfall_image2()

# j = 14212 - 3400
# ori = test.waterfall[:, test.port_len-j-1]
# test.vis_waterfall_col(ori)
# test.eval_sharpness(ori)
# deconv = test.deconv_RL_col(j, 3, 'sin2', 5, 'port')
# test.vis_waterfall_col(deconv)
# test.eval_sharpness(deconv)

print("Orz")
