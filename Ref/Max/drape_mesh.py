from auvlib.data_tools import csv_data, xtf_data
from auvlib.bathy_maps import map_draper, data_vis
import numpy as np

#Script that drapes the side-scan data on top of the generated mesh_map and
#saves the result as sss_map_images, which is a good data structure for applications.

#------------------------------------------------------------------------------
#SPECIFY PATHS AND FILENAMES

#Paths to input data
mesh_file = "../datasets/20190618_6/processed data/mesh_map.npz" #mesh_map data
sound_speeds_file = "../datasets/20190618_6/raw data/pp/SVP.txt" #sound_speeds
xtf_file = "../../datasets/20190618_6/processed data/xtf_subset2_lawnmower.cereal" #xtf data

#Filename for sss_map_images - output
outputfile_images = "sss_map_images_subset2" + ".cereal"

#Path to saving location for sss_map_images.
#If path is "" the file will be saved in the scripts location.
save_path_images = "../datasets/20190618_6/processed data/"

#------------------------------------------------------------------------------
#LOAD DATA

#Load mesh_map
mesh = np.load(mesh_file)

#Pick out the vertices, faces and bounds of the mesh_map
V, F, bounds = mesh["V"], mesh["F"], mesh["bounds"]

#Parse file or folder containing sound speed data
sound_speeds = csv_data.csv_asvp_sound_speed.parse_file(sound_speeds_file)

#Read xtf data
xtf_pings = xtf_data.xtf_sss_ping.read_data(xtf_file)

#------------------------------------------------------------------------------
#DRAPE THE MESH AND SAVE THE RESULT

#Create object used for draping
viewer = map_draper.MapDraper(V, F, xtf_pings, bounds, sound_speeds)

#Start the draping and visualize it.
viewer.show()

#Save sss_map_images
images = viewer.get_images()
map_draper.write_data(images, save_path_images + outputfile_images)
