from auvlib.data_tools import xtf_data, csv_data, all_data
from auvlib.bathy_maps import map_draper, mesh_map
import numpy as np
from tf.transformations import euler_matrix
from math import sqrt, cos, sin, acos, asin, pi, tan



''' Import Data Path '''
sound_speeds_file = "/home/chs/Desktop/Sonar/Data/SVP_max.txt"
nav_file = "/home/chs/Desktop/Sonar/Data/nav/all_attitudes.cereal"
mesh_file = "/home/chs/Desktop/Sonar/Data/EM2040/mesh.npz"

''' Load Data ''' 
mesh_data = np.load(mesh_file)
sound_speeds = csv_data.csv_asvp_sound_speed.parse_file(sound_speeds_file)
V, F, bounds = mesh_data["V"], mesh_data["F"], mesh_data["bounds"]


# all_attitudes = all_data.all_nav_attitude.read_data(nav_file) # get attitude info
# std_attitudes = all_data.convert_attitudes(all_attitudes) # convert to std attitude format

# load the side-scan
xtf_pings = xtf_data.xtf_sss_ping.read_data("/home/chs/Desktop/Sonar/Data/xtf_ping/xtf_pings_25.cereal")
#xtf_pings = xtf_data.match_attitudes(xtf_pings, std_attitudes)


# initilize a Base Draper class
viewer = map_draper.BaseDraper(V, F, bounds, sound_speeds)

# set sensor offset
r = 0.4375
viewer.set_sidescan_port_stbd_offsets(np.array([0.,r*np.cos(np.pi/4), r*(1-np.cos(np.pi/4))]) , np.array([0.,-r*np.cos(np.pi/4), r*(1-np.cos(np.pi/4))]))


# daper the 5th ping
left, right = viewer.project_ping(xtf_pings[5], 256)

# check the attributes of left and right, lots of useful stauff
meas_path = '/home/chs/Desktop/Sonar/Data/drape_result/low/meas_data_25.cereal'
meas_imgs = map_draper.sss_meas_data.read_single(meas_path)
avu_pos = meas_imgs.pos[5]
rpy = meas_imgs.rpy[5]

roll, pitch, yaw  = xtf_pings[5].roll_, xtf_pings[5].pitch_, xtf_pings[5].heading_
r = 0.4375
d_y = -r*cos(pi/4) # add difference for port and stbd
d_z = r*(1-cos(pi/4))
r_matrix = euler_matrix(0, pitch, yaw) # TODO: ignore roll?

offset = np.dot(r_matrix[:3,:3], np.array([0, d_y, d_z]))
sensor_pos = avu_pos + offset

print("Orz")