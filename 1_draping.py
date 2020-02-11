#!/usr/bin/env python

from auvlib.data_tools import std_data, csv_data, xtf_data, all_data
from auvlib.bathy_maps import mesh_map, map_draper
import numpy as np

from math import cos, pi

class MapImageSaver(object):

    def __init__(self):
        self.nbr_map_images = 0

    def save_callback(self, map_image):
        print "Got sss map image callback!"
        ''' Output Data Path '''
        filename = "meas_data_%d.cereal" % meas_list[LAYER][self.nbr_map_images]
        save_path = "/home/chs/Desktop/Sonar/Data/drape_result/"+name[LAYER]+"/" # Saving location for sss_meas_data.
        map_draper.write_data(map_image, save_path + filename)
        self.nbr_map_images += 1


''' Import Data Path '''
sound_speeds_file = "/home/chs/Desktop/Sonar/Data/SVP_max.txt"
xtf_file = "/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_%d.cereal" # sidescan data
nav_file = "/home/chs/Desktop/Sonar/Data/nav/all_attitudes.cereal"
mesh_file = "/home/chs/Desktop/Sonar/Data/EM2040/mesh.npz"

LAYER = 0 # 0:high 1:mid 2:low
name = ['high','mid','low']
meas_list = [[33,13,9,30,43,44,24,14,20],[28,36,42,26,16,41,6,38,45],[25,11,21,29,1,19,18,5,17]] # idx of swathes base on timestamp

''' Load Data ''' 
mesh_data = np.load(mesh_file)
sound_speeds = csv_data.csv_asvp_sound_speed.parse_file(sound_speeds_file)
xtf_pings = []
for idx in meas_list[LAYER]:
    xtf_pings+=xtf_data.xtf_sss_ping.read_data(xtf_file % idx)

# add attitude info
all_attitudes = all_data.all_nav_attitude.read_data(nav_file) # get attitude info
std_attitudes = all_data.convert_attitudes(all_attitudes) # convert to std attitude format
xtf_pings = xtf_data.match_attitudes(xtf_pings, std_attitudes) # put roll and pitch of attitudes in xtf_pings

''' Make Mesh from std data '''
V, F, bounds = mesh_data["V"], mesh_data["F"], mesh_data["bounds"]
# mesh_map.show_mesh(V, F)

''' Drape and save as sss_meas_data'''
# Drape and get hitpoints
saver = MapImageSaver()
viewer = map_draper.MeasDataDraper(V, F, xtf_pings, bounds, sound_speeds) # this is the one you use to get sss_meas_data
viewer.set_ray_tracing_enabled(False)
# set sensor offset
r = 0.4375
viewer.set_sidescan_port_stbd_offsets(np.array([0.,r*cos(pi/4),r*(1-cos(pi/4))]), np.array([0.,-r*cos(pi/4),r*(1-cos(pi/4))]))
viewer.set_tracing_map_size(700.) # this sets the local ray tracing environment to be 700mx700m which seems to be required...
viewer.set_image_callback(saver.save_callback) # this is called every time a track is done
viewer.set_store_map_images(False) # do not store the all the results while running, instead save them in MapImageSaver
viewer.show()

#xtf_data.show_waterfall_image(xtf_pings)