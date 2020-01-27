from auvlib.data_tools import xtf_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

xtf_file = "/home/chs/Desktop/Sonar/Data/xtf_ping/new_cropped_xtf_pings_centered.cereal" # sidescan data
xtf_pings = xtf_data.xtf_sss_ping.read_data(xtf_file)

# choose region with in mesh map
range_x = [0,1000]
range_y = [0,1000]
range_z =  [-61,-59]
''' [-45.75,-44.75]high [-61,-59]mid [-74,-62]low'''

choosen_pings = []
for ping in xtf_pings:
    if range_x[0]<ping.pos_[0]< range_x[1] and range_y[0]<ping.pos_[1]<range_y[1]and range_z[0]<ping.pos_[2]<range_z[1]:
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

''' Move vechile pos to fit std centered base '''
std_pings_moved = []
for ping in choosen_pings:
    new_pos = ping.pos_ + np.array([845.22058245, 82.73684846, 0.])
    ping.pos_ = new_pos
    std_pings_moved.append(ping)
        
#xtf_data.write_data(std_pings_moved, "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_low.cereal")
print('Num of pings: %i' % len(std_pings_moved))
