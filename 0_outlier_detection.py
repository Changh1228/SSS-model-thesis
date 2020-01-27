#!/usr/bin/env python

from auvlib.data_tools import std_data, all_data
from auvlib.bathy_maps import mesh_map
import numpy as np
import matplotlib.pyplot as plt
import copy

std_pings = std_data.mbes_ping.read_data("Data/EM2040/mid/pings_corrected.cereal")

choosen_pings = std_pings
beams = np.array(choosen_pings[0].beams)[:,2]

# set a init fit model (if can't find a good fit in first ping)
paras = np.polyfit(np.arange(np.size(beams)), beams, 8) # possible to try different order
beams_poly = np.poly1d(paras)

flag = 0 # flag for choosing model

for idx, ping in enumerate(choosen_pings):
    beams = np.array(ping.beams)[:,2]

    ''' Check noize and fix the data '''
    # pick a good fix as model for next 10 pings
    if flag < 10: # do not refresh model in 10 pings
        flag += 1
    else:
        # fit a choosen model 
        paras = np.polyfit(np.arange(np.size(beams)), beams, 8) # can try different polynomial
        beams_poly_new = np.poly1d(paras)
        varFit = np.var(beams_poly_new(np.arange(len(beams))) - beams)
        # check the variance of actuall and prediction
        if varFit < 0.01: # if a bad fit, check next ping and its fit
            beams_poly = beams_poly_new # refresh predict beams
            flag = 0

            ''' Plot predict curve'''
            # plt.figure(0)
            # plt.clf()
            # plt.title('fit: %f' % varFit)
            # plt.axis([0, 400, -95, -75])
            # plt.scatter(np.arange(len(beams)), beams, s=2, c='b', marker='.')
            # plt.plot(np.arange(len(beams)), beams_poly(np.arange(len(beams))), "red")
            # plt.pause(0.001)

    ''' Pick good beams '''
    diff = np.abs(beams_poly(np.arange(len(beams))) - beams)
    beam_idx_list = np.where(diff > 0.35)[0]
    if len(beam_idx_list) > 0:
        # discard bad beams
        choosen_pings[idx].beams = np.delete(ping.beams, beam_idx_list, axis = 0)
        # newPing is used to plot data
    #     new_ping = copy.copy(beams)
    #     new_ping[beam_idx_list] = 0 # fill bad beam with empty value(0)
    # else:
    #     new_ping = beams


    ''' Show beams before and after '''
    # plt.figure(1)
    # plt.clf()
    # plt.subplot(121)
    # plt.title('var: %f' % np.var(beams))
    # plt.axis([0, 400, -95, -75])
    # plt.grid()
    # plt.scatter(np.arange(len(beams)), beams, s=2, c='b', marker='.')
    # plt.subplot(122)
    # plt.title(idx) 
    # plt.scatter(np.arange(len(beams)), new_ping, s=2, c='b', marker='.')
    # plt.axis([0, 400, -95, -75])
    # plt.grid()
    # plt.pause(0.001)

std_data.write_data(choosen_pings, 'Data/EM2040/mid/pings_outlier_discard.cereal')

''' Show mesh (praying Orz...) '''
V_f, F_f, bounds_f = mesh_map.mesh_from_pings(choosen_pings, 0.5)
mesh_map.show_mesh(V_f, F_f)