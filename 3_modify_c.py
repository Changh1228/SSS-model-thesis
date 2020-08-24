import numpy as np
import matplotlib.pyplot as plt


def smooth(x, window_len=20, window='hanning'):

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

plt.subplot(1,2,1)
correction = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor_beam_angle-theta.csv") , delimiter=",")
plt.scatter(correction[:,0], correction[:,1], s=2, c='b', marker='.')
plt.ylim(0,4)
plt.subplot(1,2,2)
plt.ylim(0,4)
correction = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor_beam_angle-theta.csv") , delimiter=",")
correction[:,1] =smooth(correction[:,1])
plt.scatter(correction[:,0], correction[:,1], s=2, c='b', marker='.')
plt.show()
np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/factor_beam_angle-theta_smoothed.csv", correction, delimiter=',')


# correction_low = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor_distance-r_low.csv") , delimiter=",")
# plt.scatter(correction_low[:,0], correction_low[:,1], s=2, c='b', marker='.')
# correction_mid = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor_distance-r_mid.csv") , delimiter=",")
# plt.scatter(correction_mid[:,0], correction_mid[:,1], s=2, c='r', marker='.')
# correction_mid[:,1] = (correction_mid[:,1] + correction_low[:,1])/2
# correction_mid[:,1] =smooth(correction_mid[:,1])
# plt.scatter(correction_mid[:,0], correction_mid[:,1], s=2, c='r', marker='.')
# plt.show()
# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/factor_distance-r.csv", correction_mid, delimiter=',')