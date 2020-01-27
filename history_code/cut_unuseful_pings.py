from auvlib.data_tools import xtf_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

''' cut some unuseful pings in each layer '''

xtf_file = "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_low.cereal" # sidescan data
xtf_pings = xtf_data.xtf_sss_ping.read_data(xtf_file)

choosen_pings = []
for ping in xtf_pings:
    if 1560597112120 < ping.time_stamp_ <  1560598940700:
        choosen_pings.append(ping)
''' Show moving curve of vechile in xtf ''' 
x_xtf = []
y_xtf = []
z_xtf = []
for ping in choosen_pings:
    x_xtf.append(ping.pos_[0])
    y_xtf.append(ping.pos_[1])
    z_xtf.append(ping.pos_[2])
ax = plt.subplot(projection='3d')
ax.scatter(x_xtf, y_xtf, z_xtf, s=2, c='r', marker='.')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

xtf_data.write_data(choosen_pings, "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_low_1.cereal")
print('Num of pings: %i' % len(choosen_pings))