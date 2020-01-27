#!/usr/bin/env python

from auvlib.data_tools import std_data, all_data
from auvlib.bathy_maps import mesh_map
import numpy as np
import matplotlib.pyplot as plt
import copy

# std_pings = std_data.mbes_ping.read_data("/home/chs/Desktop/Sonar/Data/KTH1/submaps.cereal")

file_path = "/home/chs/Desktop/KTH1"

# read all_mbes_ping from .all data
all_ping = all_data.all_mbes_ping.parse_folder(file_path)

# read all_nav_entry from .all data
nav = all_data.all_nav_entry.parse_folder(file_path)

# convert .all file to std data
std_pings = all_data.convert_matched_entries(all_ping, nav)

choosen_pings = std_pings

''' check the incontinous in one ping '''
new_pings  = []
beams_buff = np.array(choosen_pings[0].beams)[:,2]
for idx, ping in enumerate(choosen_pings):
    beams = np.array(ping.beams)[:,2]

    beams_left = copy.copy(beams)
    beams_left[0:-1] = beams[1:] # move beams one index to left direction
    delta = abs(beams_left - beams)

    # check the inconsistincy in one ping
    beam_idx_list = np.where(delta > 1)[0]

    if len(beam_idx_list) > 0: # "make up" new ping from previous ping
        vechile_move = ping.pos_ - new_pings[-1].pos_
        new_beams = np.array(new_pings[-1].beams) + vechile_move
        ping.beams = new_beams
        new_pings.append(ping)
    else:
        new_pings.append(ping)
        beams_buff = beams

std_data.write_data(new_pings, '/home/chs/Desktop/KTH1/pings_outlier_fixed.cereal')

''' Show mesh (praying Orz...) '''
V_f, F_f, bounds_f = mesh_map.mesh_from_pings(new_pings, 2.0)
mesh_map.show_mesh(V_f, F_f)