from auvlib.data_tools import std_data, all_data
from auvlib.bathy_maps import mesh_map
import numpy as np
from matplotlib import pyplot as plt

std_pings = std_data.mbes_ping.read_data("../datasets/20190618_6/processed data/mbes_lawnmower_pings.cereal")

beams = []
size_list = []

for ping in std_pings:
    beam = np.array(ping.beams)[:,2]
    if np.size(beam) > 0:
        beams.append(beam)
        size_list.append(np.size(beam))

params = []
for beam in beams:
    params.append(np.polyfit(np.arange(np.size(beam)), beam, 2))

predicted_beams = []
polynomials = []

for param_set in params:
    polynomials.append(np.poly1d(param_set))

for i in range(len(size_list)):
    print(i)
    polynomial = polynomials[i]
    size = size_list[i]
    beam = beams[i]

    difference = np.abs(polynomial(np.arange(size)) - beam)
    idx_list = np.where(difference > 0.3)[0]

    if len(idx_list) != 0:
        new_beam = []
        for idx, elem in enumerate(std_pings[i].beams):
            if idx not in idx_list:
                new_beam.append(elem)
        std_pings[i].beams = new_beam

V, F, bounds = mesh_map.mesh_from_pings(std_pings, 0.5)
mesh_map.show_mesh(V, F)

np.savez("beautiful_mesh.npz", V=V, F=F, bounds = bounds)
