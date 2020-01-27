from auvlib.data_tools import xtf_data, all_data, gsf_data, std_data
from auvlib.bathy_maps import mesh_map
import numpy as np
import os

#Script that parses folders of xtf and multibeam data and saves it as .cereal files.

#------------------------------------------------------------------------------
#SPECIFY PATHS AND FILENAMES

#Paths to folders with sonar data
xtf_folder = "../datasets/20190618_6/raw_data/pp/XTF-data/subset1" #Directory with xtf data
mbes_folder = "../datasets/20190618_6/raw_data/pp/EM2040" #Directory with multibeam data

#Filenames - output
outputfile_xtf = "xtf_prep_test" + ".cereal" #Filename output xtf file
outputfile_mbes = "mbes_prep_test" + ".cereal" #Filename output multibeam file

#Paths to saving location. If path is "" the file will be saved in the scripts location.
save_path_xtf = "../datasets/20190618_6/processed data/" #xtf
save_path_mbes = "../datasets/20190618_6/processed data/" #multibeam

#------------------------------------------------------------------------------
#PREPARE XTF DATA

#Convert xtf data to .cereal file and save it
xtf_pings = xtf_data.xtf_sss_ping.parse_folder(xtf_folder)
xtf_data.write_data(xtf_pings, save_path_xtf + outputfile_xtf)


#------------------------------------------------------------------------------
#PREPARE MULTIBEAM DATA

#Decide if multibeam folder contains .all files or .gsf files and convert to std_data
for r, d, f in os.walk(mbes_folder): # r = root, d = directory, f = files
    for file in f:
        if ".all" in file:
            all_pings = all_data.all_mbes_ping.parse_folder(mbes_folder)
            all_entries = all_data.all_nav_entry.parse_folder(mbes_folder)

            #Convert to standard data type for working with multibeam data (std_data)
            std_pings = all_data.convert_matched_entries(all_pings, all_entries)
            break
        elif ".gsf" in file:
            mbes_pings = gsf_data.gsf_mbes_ping.parse_folder(mbes_folder)

            #Convert to standard data type for working with multibeam data (std_data)
            std_pings = gsf_data.convert_pings(mbes_pings)
            break
    else:
        #Continue if inner loop wasn't broken
        continue
    #If inner loop was broken, break outer loop
    break

#Write multibeam data to file and save it to the specified location
std_data.write_data(std_pings, save_path_mbes + outputfile_mbes)
