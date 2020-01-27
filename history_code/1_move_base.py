from auvlib.data_tools import xtf_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

xtf_file = "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_mid.cereal" # sidescan data
new_xtf_pings = xtf_data.xtf_sss_ping.read_data(xtf_file)

xtf_file = "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_mid_old.cereal" # sidescan data
old_xtf_pings = xtf_data.xtf_sss_ping.read_data(xtf_file)

for ping in old_xtf_pings:
    if new_xtf_pings[0].time_stamp_ == ping.time_stamp_:
        delta_pos = new_xtf_pings[0].pos_ - ping.pos_
        print("Delta pos =")
        print(delta_pos)

''' Delta pos = [845.22058245  82.73684846    0. ]'''
''' Note: this step has been added to divide data '''
std_pings_moved = []
for ping in new_xtf_pings:
    new_pos = ping.pos_ + np.array([845.22058245, 82.73684846, 0.])
    ping.pos_ = new_pos
    std_pings_moved.append(ping)

xtf_data.write_data(std_pings_moved, "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_mid_new_1.cereal")
print("???")