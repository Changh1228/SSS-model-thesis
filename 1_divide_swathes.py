from auvlib.data_tools import xtf_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

''' Divide xtf data into swathes'''
def divide_swathes():
    xtf_file = "/home/chs/Desktop/Sonar/Data/xtf_ping/new_cropped_xtf_pings_centered.cereal" # sidescan data
    xtf_pings = xtf_data.xtf_sss_ping.read_data(xtf_file)
    i = 0
    id_start = 0
    time = []
    for idx, ping in enumerate(xtf_pings): # ignore the first file header
        if ping.first_in_file_ and idx != 0: # fing next file header (idx-1=end of file)
			#print(xtf_pings[])
            choosen_pings = xtf_pings[id_start:idx]
            time.append([i, saver(choosen_pings, i)])           
            id_start = idx # save the id of header
            i += 1
    # save the last file
    choosen_pings = xtf_pings[id_start:]
    time.append([i, saver(choosen_pings, i)])
    time = np.array(time)
    print(time[np.lexsort(time.T)])

def saver(choosen_pings, i):
    choosen_pings =move_base(choosen_pings)
    xtf_data.write_data(choosen_pings, "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_%d.cereal" % i)
    # print('Num of pings: %i' % len(choosen_pings))
    return choosen_pings[0].time_stamp_



def move_base(choosen_pings):
    # move base of pings
    xtf_pings_moved = []
    for ping in choosen_pings:
        new_pos = ping.pos_ + np.array([845.22058245, 82.73684846, 0.])
        ping.pos_ = new_pos
        xtf_pings_moved.append(ping)
    return xtf_pings_moved

def print_swathes(choosen_pings):
    x_xtf = []
    y_xtf = []
    z_xtf = []
    for ping in choosen_pings:
        x_xtf.append(ping.pos_[0])
        y_xtf.append(ping.pos_[1])
        z_xtf.append(ping.pos_[2])
    ax = plt.subplot(projection='3d')
    ax.scatter(x_xtf, y_xtf, z_xtf, s=2, c='b', marker='.')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

# divide_swathes()

''' Visualize route of vechile to decide layers '''
def decide_layer():
    idxs = [28,36,42,26,16,41,6,38,45]# index in xtf_pings, list in time_stamp
    #[12,27,32,40,35,37,46,3,2,15,8,10,34,0,33,13,9,30,43,44,24,14,20,23,4,28,36,42,26,16,41,6,38,45,39,25,11,21,29,1,19,18,5,17,7,31,22]
    x_xtf = []
    y_xtf = []
    z_xtf = []
    for idx in idxs:
        xtf_pings = xtf_data.xtf_sss_ping.read_data("/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_%d.cereal" % idx)
        print(idx)
        for ping in xtf_pings:
            x_xtf.append(ping.pos_[0])
            y_xtf.append(ping.pos_[1])
            z_xtf.append(ping.pos_[2])
        ax = plt.subplot(projection='3d')
        ax.scatter(x_xtf, y_xtf, z_xtf, s=2, c='r', marker='.')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        plt.pause(0.1)
    plt.show()
        


decide_layer()
#divide_swathes()