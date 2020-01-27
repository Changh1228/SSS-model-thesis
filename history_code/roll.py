#!/usr/bin/env python

from auvlib.data_tools import std_data, all_data
from auvlib.bathy_maps import mesh_map
import numpy as np
import math
import copy
from tf.transformations import euler_matrix


class nav:
    def __init__(self, timestamps, yaws, rolls, pitchs):
        self.timestamps = timestamps
        self.yaws = yaws
        self.rolls = rolls
        self.pitchs = pitchs

def calibrate(nav, pings, phi):
    # nav: data from head.txt. pings: multibeam data. phi: hand fixed angle
    # Calibrate beam using nav data
    k = 0
    for ping in pings:
        time_str = str(ping.time_stamp_)[0:10]
        index = [i for i,x in enumerate(nav.timestamps) if x.find(time_str)!=-1]
        if len(index) != 1:
            print("Can't find")
            print(index.__len__)
            continue
        else:
            # Get rotation matrix from euler angle
            # Simplify: ignore the timestamp after num dot
            # todo: if simple methord works, using ration methord in AUVlib
            index = index[0]
            _RadRoll =  nav.rolls[index] * math.pi / 180 + phi # fix roll data by hand at here
            _RadPitch = nav.pitchs[index] * math.pi / 180
            _RadYaw = nav.yaws[index] * math.pi / 180
            R = euler_matrix(_RadRoll, _RadPitch, _RadYaw)
            R = R[0:3]
            R = R[:,0:3]

            Rz = euler_matrix(0, 0, _RadYaw)
            Rz = Rz[0:3]
            Rz = Rz[:,0:3]

            # correct beam
            new_beams = []
            for beam in ping.beams:
                beam = np.dot(Rz.T, (beam-ping.pos_))
                beam = ping.pos_ + np.dot(R, beam)
                new_beams.append(beam)
            pings[k].beams = new_beams
            k += 1

    return pings

# read multibeam data
std_pings = std_data.mbes_ping.read_data("Data/std_pings_centered.cereal")

# read nav data 
f = open("Data/nav/head.txt")
line = f.readline()
timestamps = []
rolls = []
pitchs = []
yaws = []
while line:
    a = line.split()
    _timestamp = a[0:1][0]
    _yaw = float(a[4:5][0])
    _roll = float(a[5:6][0])
    _pitch = float(a[6:7][0])
    timestamps.append(_timestamp)
    yaws.append(_yaw)
    rolls.append(_roll)
    pitchs.append(_pitch)
    line = f.readline()
    head = nav(timestamps, yaws, rolls, pitchs)
f.close()

# second run 58k~71k(1st line 58K~59.7K, 2nd line 59.8k~61.3k, 3rd line 61.5k~62.9k, 4th line 63.2k~64.6k)
one_line = []+std_pings[58000:59500]
two_line = []+std_pings[59900:61150]
thr_line = []+std_pings[62000:62800]#[]+std_pings[61550:62800]
fou_line = []+std_pings[63250:64500]

one_line_new = calibrate(head, one_line, 0.06)
two_line_new = calibrate(head, two_line, 0.06)
thr_line_new = calibrate(head, thr_line, -0.03)

new_pings = one_line_new + two_line_new + thr_line_new# + fou_line

# Show mech (praying)
V_f, F_f, bounds_f = mesh_map.mesh_from_pings(new_pings, 0.5) # .5 is the resolution of the constructed mesh
mesh_map.show_mesh(V_f, F_f)
