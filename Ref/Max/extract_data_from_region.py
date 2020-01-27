from auvlib.data_tools import std_data, xtf_data

#Script that reads multibeam and xtf_data from .cereal files and extracts relevant data
#in a region of the users choice. Saves mesh_map of selected region as well as multibeam
#and xtf_data originating from that region.

#------------------------------------------------------------------------------
#SPECIFY PATHS, FILENAMES, REGION BOUNDS AND MESH_MAP RESOLUTION

#Path to .cereal files with sonar data
xtf_file = "../datasets/20190618_6/processed data/xtf_subset1.cereal" #.cereal file with xtf data
mbes_file = "../datasets/20190618_6/processed data/corrected_mbes.cereal" #.cereal file with multibeam data

#Filenames - output
outputfile_xtf = "xtf_lawnmover_test" + ".cereal" #Filename output xtf file
outputfile_mbes = "mbes_lawnmower_test" + ".cereal" #Filename output multibeam file

#Paths to saving location. If path is "" the file will be saved in the scripts location.
save_path_xtf = "../datasets/20190618_6/processed data/" #xtf
save_path_mbes = "../datasets/20190618_6/processed data/" #multibeam

#Specify region bounds.
#To find reasonable region bounds, use generate_mesh_map.py for visulization.
low_x = 650412
high_x = 652245
low_y = 0
high_y = 6471750

#------------------------------------------------------------------------------
#SAVE XTF DATA FROM SELECTED REGION

#Read xtf data
xtf_data = xtf_data.xtf_sss_ping.read_data(xtf_file)

#Pick out data points from selected region
xtf_region_data = []
for ping in xtf_data:
    pos = ping.pos_
    if pos[0] > low_x and pos[0] < high_x and pos[1] > low_y and pos[1] < high_y:
        xtf_region_data.append(ping)

#Save data points from selected region
xtf_data.write_data(xtf_region_data, save_path_xtf + outputfile_xtf)

#------------------------------------------------------------------------------
#SAVE MULTIBEAM DATA FROM SELECTED REGION

#Read multibeam data
mbes_data = std_data.mbes_ping.read_data(mbes_file)

#Pick out data points from selected region
mbes_region_data = []
for ping in mbes_data:
    pos = ping.pos_
    if pos[0] > low_x and pos[0] < high_x and pos[1] > low_y and pos[1] < high_y:
        mbes_region_data.append(ping)

#Save data points from selected region
std_data.write_data(mbes_region_data, save_path_mbes + outputfile_mbes)
