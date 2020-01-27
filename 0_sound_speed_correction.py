#!/usr/bin/env python

from auvlib.data_tools import std_data, all_data
from auvlib.bathy_maps import mesh_map
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

file_path = "/home/chs/Desktop/Sonar/Data/EM2040/mid"

# read all_mbes_ping from .all data
all_ping = all_data.all_mbes_ping.parse_folder(file_path)

# read all_nav_entry from .all data
nav = all_data.all_nav_entry.parse_folder(file_path)

# convert .all file to std data
std_pings = all_data.convert_matched_entries(all_ping, nav)


def print_ping(beams,j):
    # print one ping and title
    plt.clf()
    plt.title((j,len(beams)))
    plt.grid()
    plt.axis([0, 400, -95, -80])
    x = np.linspace(start = 1, stop = len(beams), num = len(beams))
    plt.scatter(x, beams, s=2)
    plt.pause(0.01)
    plt.show()


def func(x, a, b, c):
    return a*x**4 + b*x**2 + c

def correct_overlapping(std_pings, sample_num): 
    # sample_num: index of symmetry and flat ping (relatively, choose manually)
    ''' Cal the relation between id and angle'''
    test_ping = std_pings[sample_num]
    angle = []
    i = 0
    for beam in test_ping.beams:
        d_z = test_ping.pos_[2] - beam[2]
        d_xy = math.sqrt((test_ping.pos_[0] - beam[0])**2 + (test_ping.pos_[1] - beam[1])**2)
        if i < 200:
            rad = -math.atan2(d_xy, d_z)
        else:
            rad = math.atan2(d_xy, d_z)
        angle.append(rad)
        i += 1
    angle = np.array(angle)

    ''' Make a smooth curve of attitude '''
    sample_pings = std_pings[sample_num]
    ave_beams = np.array(std_pings[sample_num].beams)[:,2] # choose a good and flat ping (relatively)
    # i = 1
    # for ping in sample_pings:
    #     beams = np.array(ping.beams)[:,2]
    #     if len(beams) != 400:
    #         continue
    #     i += 1
    #     ave_beams += beams
    # ave_beams /= i
    

    ''' "Gaussian" smooth '''
    a = ave_beams[0:6]
    b = ave_beams[394:400]
    kernel = np.array([1,1,1,1,1,1,3,1,1,1,1,1,1]) # size 13
    ave_beams = np.convolve(ave_beams, kernel, 'valid')/15
    ave_beams = np.hstack((a, ave_beams))
    ave_beams = np.hstack((ave_beams, b))
    #print_ping(ave_beams, 'Beam Model')
    para =  max(ave_beams) / ave_beams  # para used to correct beams


    ''' Fit a curve to get better correct para'''
    a, b, c = optimize.curve_fit(func, angle, para)[0]
    a *= 1.1


    ''' Use angle and curve to correct beams '''
    choosen_pings = std_pings
    new_pings = []
    for ping in choosen_pings:
        index = 0 # id of beams
        new_beams = []
        for beam in ping.beams:
            d_z = ping.pos_[2] - beam[2]
            d_xy = math.sqrt((ping.pos_[0] - beam[0])**2 + (ping.pos_[1] - beam[1])**2)
            if index < 200:
                tan = -math.atan2(d_xy, d_z)
            else:
                tan = math.atan2(d_xy, d_z)

            new_beam = beam[0:2]
            para = func(tan, a, b, c)
            z =  para * beam[2]
            new_beam = np.hstack((new_beam, np.array([z])))
            new_beams.append(new_beam)
            index += 1
        ping.beams = new_beams
        new_pings.append(ping)

    ''' Check the result of correction from two pings'''
    # ax = plt.subplot(projection='3d')
    # ping1 = choosen_pings[13500]
    # ping2 = choosen_pings[16000]
    # x1 = np.array(ping1.beams)[:,0]
    # y1 = np.array(ping1.beams)[:,1]
    # z1 = np.array(ping1.beams)[:,2]
    # x2 = np.array(ping2.beams)[:,0]
    # y2 = np.array(ping2.beams)[:,1]
    # z2 = np.array(ping2.beams)[:,2]
    # ax.scatter(x1, y1, z1, c='r')
    # ax.scatter(x2, y2, z2, c='b') 
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.show()
    return new_pings

# choosen_pings = std_pings[200:]
# for idx, ping in enumerate(choosen_pings):
#     beams = np.array(ping.beams)[:,2]
#     if len(beams) == 400:
#         print_ping(beams,idx)


''' Run sound speed correction '''
new_pings = correct_overlapping(std_pings, 297) # 1277 for low, 2300 for mid

''' Save corrected pings as cereal file '''
std_data.write_data(std_pings, 'Data/EM2040/mid/pings_corrected.cereal')

''' Show mesh (praying Orz...) '''
V_f, F_f, bounds_f = mesh_map.mesh_from_pings(std_pings, 0.5)
mesh_map.show_mesh(V_f, F_f)

''' Find bound position '''