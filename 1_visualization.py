#!/usr/bin/env python

from auvlib.bathy_maps import map_draper
import matplotlib.pyplot as plt
import numpy as np

# read sss_map_image from cereal file
image_path = '/home/chs/Desktop/Sonar/Data/drape_result/low/meas_data_8.cereal'
image = map_draper.sss_meas_data.read_single(image_path)

# Take one image as exp

print('???')
# filled zeros with closer base
# new_depth = []
# for roll in image.sss_waterfall_depth:
#     new_roll = []
#     for pixel in roll:
#         if pixel == 0 :
#             new_roll.append(-90)
#         else:
#             new_roll.append(pixel)
#     new_depth.append(new_roll)
# new_depth = np.array(new_depth)

# plt.subplot(221)
# plt.imshow(image.sss_map_image)
# plt.title("map_image")
# plt.subplot(222)
# plt.imshow(image.sss_waterfall_image)
# plt.title('waterfall_image')
# plt.subplot(223)
# plt.imshow(new_depth)
# plt.title('waterfall_depth')
# plt.subplot(224)
# plt.imshow(image.sss_waterfall_model)
# plt.title('waterfall_model')

plt.imshow(image.sss_waterfall_image)
# plt.imshow(image.sss_waterfall_image)
# plt.imshow(new_depth)
# plt.imshow(image.sss_waterfall_model)
plt.show()