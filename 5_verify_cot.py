#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math

def smooth(x, window_len=7, window='hanning'):

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[(window_len-1)/2:0:-1],x,x[-2:(-window_len-3)/2:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def square_error(data, func):
    catch = func-data
    return np.sum(catch**2)

correction = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor_incident_angle-phi_test.csv") , delimiter=",")
correction[:,1] = smooth(correction[:,1])

# find factor range
f_cor = []
f_else = []
for point in correction:
    if point[0] < 1.4 and point[0] > 0.9:
        f_cor.append(point)
    else:
        f_else.append(point)
f_else = np.array(f_else)
f_cor = np.array(f_cor)


# find best scaling parameters
k_cot = np.sum(f_cor[:,1])/np.sum(abs(np.tan(f_cor[:,0])))
k_cos = np.sum(f_cor[:,1])/np.sum(abs(1/np.cos(f_cor[:,0])))
k_cos2 = np.sum(f_cor[:,1])/np.sum(1/np.cos(f_cor[:,0])**2)

cot = abs(np.tan(f_cor[:,0])) * k_cot
cos = abs(1/np.cos(f_cor[:,0])) * k_cos
cos2 = 1/np.cos(f_cor[:,0])**2 * k_cos2

print("cot fit: %f" % (square_error(f_cor[:,1], cot)))
print("cos fit: %f" % (square_error(f_cor[:,1], cos)))
print("cos2 fit: %f" % (square_error(f_cor[:,1], cos2)))

#exit()
plt.ylim(0,3)
plt.xlim(-1.5,1.5)
plt.xlabel('Incidence angle/rad', fontsize=17)
plt.ylabel('$C(\phi)$', fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
#plt.xlim(-1.7,-0.6)
plt.scatter(f_cor[:,0], f_cor[:,1], s=80, c='k', marker='.')
plt.scatter(f_else[:,0], f_else[:,1], s=20, c='gray', marker='.')
plt.plot(correction[:,0], abs(np.tan(correction[:,0])) * k_cot, c='r', label = '$1/cot$',linewidth=3)
plt.plot(correction[:,0], abs(1/np.cos(correction[:,0])* k_cos), c='g', label = '$1/cos$', ls = '--', linewidth=3)
plt.plot(correction[:,0], abs(1/np.cos(correction[:,0])**2*k_cos2), c='b', label = '$1/cos^2$', ls=':', linewidth=3)
#plt.title("RMSE fitting cos^2 = 0.2567948679196528")
plt.legend(fontsize=17)
plt.show()

print("???")
