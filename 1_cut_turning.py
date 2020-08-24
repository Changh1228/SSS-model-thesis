from auvlib.data_tools import xtf_data
from auvlib.bathy_maps import map_draper
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

LAYER =2 # 0:high 1:mid 2:low
name = ['high','mid','low']
meas_list = [[33,13,9,30,43,44,24,14,20],[28,36,42,26,16,41,6,38,45],[25,11,21,29,1,19,18,5,17]]

def print_swathes():
    x_xtf = []
    y_xtf = []
    z_xtf = []
    for idx in meas_list[LAYER]:
        meas_path = '/home/chs/Desktop/Sonar/Data/drape_result/'+name[LAYER]+'/meas_data_%d.cereal' % idx
        meas_imgs = map_draper.sss_meas_data.read_single(meas_path)
        for pos in meas_imgs.pos[100:-100]:
            x_xtf.append(pos[0])
            y_xtf.append(pos[1])
            z_xtf.append(pos[2])
    plt.figure(0, figsize=(10,5))
    plt.axis('equal')  
    ax = plt.subplot(1,2,1,projection='3d')
    ax.scatter(x_xtf, y_xtf, z_xtf, s=2, c='b', marker='.')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.subplot(1,2,2)
    plt.scatter(x_xtf, y_xtf, s=2)
    plt.show()



print_swathes()

print('Orz')