#!/usr/bin/env python

# Hongsheng Chang
# changh@kth.se
# 2019.06.28
# Referemce: https://gist.github.com/nilsbore/3b0e4c608a06d6950cd649cfc22201a7 https://nilsbore.github.io/auvlib-docs/
# Code for Sonar Calibration in SMaRC

from auvlib.data_tools import xtf_data, std_data, all_data
from auvlib.bathy_maps import mesh_map
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 1 Multi-beam data
std_pings = std_data.mbes_ping.read_data("Data/std_pings_centered.cereal")

# print "Total number of sidescan pings:", len(std_pings)
# print "Position of sensor:", std_ping0.pos_
# print "Number of multibeam hits:", len(std_ping0.beams)
# print "Position of first hit:", std_ping0.beams[0]
# print "Time of data collection:", std_ping0.time_string_

# read nav data
nav = all_data.all_nav_attitude.read_data("Data/all_attitudes.cereal")

# match attitude
#new_pings = all_data.match_attitude(std_pings, nav)

# Show moving curve
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# X = []
# Y = []
# Z = []

# for ping in std_pings:
#     X.append(ping.pos_[0])
#     Y.append(ping.pos_[1])
#     Z.append(ping.pos_[2])

# range_l = 58000
# range_h = 71000
# ax.scatter(X[range_l:range_h], Y[range_l:range_h], Z[range_l:range_h])
# plt.show()

# # first run 47k~57k
# # second run 58k~71k(1st line 58K~59.7K, 2nd line 59.8k~61.3k, 3rd line 61.3k~62.9k, 4th line 63.2k~64.6k)

# Mesh map
V_f, F_f, bounds_f = mesh_map.mesh_from_pings(std_pings[61500:64600], 0.5) # .5 is the resolution of the constructed mesh
mesh_map.show_mesh(V_f, F_f)

# 2 Sidescan data 
# xtf_pings = xtf_data.xtf_sss_ping.read_data("xtf_pings_centered.cereal")
# xtf_ping0 = xtf_pings[0] # let's just look at the first ping!
# print "Total number of sidescan pings:", len(xtf_pings)
# print "Position of sensor:", xtf_ping0.pos_
# print "Number of port sidescan intensities:", len(xtf_ping0.port.pings)
# print "Number of starboard sidescan intensities:", len(xtf_ping0.stbd.pings)
# print "Time of data collection:", xtf_ping0.time_string_
# print "Intensity of first port hit (time 0s):", xtf_ping0.port.pings[0]
# print "Time of arrival of last port intensity (s):", xtf_ping0.port.time_duration
# print "Intensity of last port hit:", xtf_ping0.port.pings[-1]