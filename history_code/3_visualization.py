#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import tan

data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_beam_angle.csv"), delimiter=",")

''' 3D beam-incidence-intensity '''
# index = 0
# while index < len(data):
#     if index %1000 ==0:
#         print("row %d/%d done" % (index, len(data)))
#     row = data[index]
#     key = np.where((data[:,2:4]==row[2:4]).all(1))[0]
#     value = data[key,4]
#     data[index,4] = np.mean(value)
#     data = np.delete(data, key[1:],axis=0)
#     index += 1
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# beam_angle = data[:,2]
# incidence = data[:,3]
# indensity = data[:,4]
# ax.scatter(beam_angle, incidence, indensity)
# plt.show()

''' 2D beam or incidence-intensity'''
index = 0
while index < len(data):
    if index %100 ==0:
        print("row %d/%d done" % (index, len(data)))
    row = data[index]
    key = np.where((data[:,1:2]==row[1:2]).all(1))[0]
    value = data[key,4]
    data[index,4] = np.mean(value)
    data = np.delete(data, key[1:],axis=0)
    index += 1
beam_angle = data[:,2]
incidence = data[:,3]
indensity = data[:,4]
distance = data[:,1]
plt.figure(0)
plt.grid()
plt.title("beam_angle-intensity")
plt.scatter(beam_angle, indensity, s=2, c='b', marker='.')
plt.show()

''' Beam pattern '''
# index = 0
# while index < len(data):
#     if index %1000 ==0:
#         print("row %d/%d done" % (index, len(data)))
#     row = data[index]
#     key = np.where((data[:,2:3]==row[2:3]).all(1))[0]
#     value = data[key,4]
#     data[index,4] = np.mean(value) * abs(tan(row[3]))
#     data = np.delete(data, key[1:],axis=0)
#     index += 1

# ax = plt.subplot(111)#, projection='polar')
# #ax.set_rlim(0,1.0)
# angle = data[:,2]
# beam = data[:,4]
# ax.scatter(angle, beam, s=2, c='b')
# plt.show()