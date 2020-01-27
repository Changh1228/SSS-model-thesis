from auvlib.data_tools import std_data
from auvlib.bathy_maps import mesh_map
import numpy as np

#Script that creates and saves mesh_map from multibeam data and optionally
#visualizes it.

#------------------------------------------------------------------------------
#SPECIFY PATHS, FILENAMES, MESH_MAP RESOLUTION AND IF YOU WANT TO VISUALIZE THE RESULT

#Path to .cereal files with multibeam data
mbes_file = "../datasets/20190618_6/processed data/mbes_lawnmower_pings_corrected.cereal"

#Filename for the mesh_map - output
outputfile_mesh_map = "mesh_map" + ".npz"

#Path to saving location for mesh_map.
#If path is "" the file will be saved in the scripts location.
save_path_mesh = "../datasets/20190618_6/processed data/"

#Specify mesh_map resolution
resolution = .5

#Visualize the result?
visualization = True

#------------------------------------------------------------------------------
#LOAD MULTIBEAM DATA AND SAVE MESH_MAP FROM SELECTED REGION

#Load multibeam data
mbes_data = std_data.mbes_ping.read_data(mbes_file)
mbes_data = mbes_data[60000:70000]

#Create mesh_map. V = vertices, F = faces, bounds = bounds
V, F, bounds = mesh_map.mesh_from_pings(mbes_data, resolution)

#Save mesh_map
np.savez(save_path_mesh + outputfile_mesh_map, V=V, F=F, bounds = bounds)

#------------------------------------------------------------------------------
#VISUALIZE MESH_MAP (OPTIONAL)

if visualization:
    mesh_map.show_mesh(V,F)
