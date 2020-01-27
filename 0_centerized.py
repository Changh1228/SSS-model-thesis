#!/usr/bin/env python

from auvlib.data_tools import xtf_data, std_data, all_data
from auvlib.bathy_maps import mesh_map
import numpy as np


# # Load centered data
# std_pings_centered = std_data.mbes_ping.read_data("Data/std_pings_centered.cereal")

# # Load raw data .all file
# file_path = "/home/chs/Desktop/Sonar/Data/EM2040/low"

# # read all_mbes_ping from .all data
# all_ping = all_data.all_mbes_ping.parse_folder(file_path)

# # read all_nav_entry from .all data
# nav = all_data.all_nav_entry.parse_folder(file_path)

# # convert .all file to std data
# std_pings = all_data.convert_matched_entries(all_ping, nav)

# for ping in std_pings_centered:
#     if std_pings[0].time_stamp_ == ping.time_stamp_:
#         delta_pos = std_pings[0].pos_ - ping.pos_
#         print("Delta pos =")
#         print(delta_pos)
        

''' After checking, the delta pos is 650461.68011388 6471645.21805158'''

# Load Uncented pings
std_pings = std_data.mbes_ping.read_data("Data/EM2040/mid/pings_outlier_discard.cereal")
std_pings_centered = []
for ping in std_pings:
    new_pos = ping.pos_ - np.array([650461.68011388, 6471645.21805158, 0])
    ping.pos_ = new_pos
    new_beams = []
    for beam in ping.beams:
        new_beam = beam - [650461.68011388, 6471645.21805158, 0]
        new_beams.append(new_beam)
    ping.beams = new_beams
    std_pings_centered.append(ping)

std_data.write_data(std_pings_centered, 'Data/EM2040/mid/pings_centered.cereal')
print("???")
