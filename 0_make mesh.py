#!/usr/bin/env python

from auvlib.data_tools import std_data
from auvlib.bathy_maps import mesh_map
import numpy as np

''' mix std pings from mid run and low run and make a mesh fro draping '''

low_pings = std_data.mbes_ping.read_data("Data/EM2040/low/pings_centered.cereal")
mid_pings = std_data.mbes_ping.read_data("Data/EM2040/mid/pings_centered.cereal")

choosen_pings = low_pings + mid_pings
V, F, bounds = mesh_map.mesh_from_pings(choosen_pings, 0.5)
mesh_map.show_mesh(V, F)

np.savez("Data/EM2040/mesh.npz", V= V, F=F, bounds=bounds)
print("???")