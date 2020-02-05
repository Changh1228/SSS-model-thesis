#!/usr/bin/env python

from auvlib.data_tools import std_data
from auvlib.bathy_maps import mesh_map, map_draper
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, acos, asin, pi, tan
import time


class beam:
    def __init__(self,auv_index, beam_index, meas_imgs, bounds, N, mesh_reso):
        self.vechile_pos = meas_imgs.pos[auv_index] # [X, Y, Z]
        self.beam_pos = np.array([meas_imgs.sss_waterfall_hits_X[auv_index, beam_index], 
                                  meas_imgs.sss_waterfall_hits_Y[auv_index, beam_index], 
                                  meas_imgs.sss_waterfall_hits_Z[auv_index, beam_index]])
        self.incident_angle, self.beam_angle, self.dist, self.deltaz = self.cal_triangle(auv_index, beam_index, bounds, N, mesh_reso)
        self.intensity = meas_imgs.sss_waterfall_image[auv_index, beam_index]
        

    def cal_triangle(self, auv_index, beam_index,bounds, N, mesh_reso):
        # bounds [[xmin, ymin],[xmax, ymax]] of mesh; N normal of whole mesh
        roll  = meas_imgs.rpy[auv_index][0]
        pitch = meas_imgs.rpy[auv_index][1]
        yaw   = meas_imgs.rpy[auv_index][2]
        auv_vec = np.array([cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]) # unit vector of AUV pose
        beam_vec = self.vechile_pos - self.beam_pos # vector from hit point to AUV
        beam_dist = np.linalg.norm(beam_vec)
        ''' cal beam angle '''
        angle = asin(beam_vec[2]/beam_dist)# angle to level,abs
        direction = np.cross(beam_vec[0:2], auv_vec[0:2]) # decide +-
        angle  = angle *direction/abs(direction)
        angle -= meas_imgs.rpy[auv_index][0] # fix with roll
        ''' get normal vector at hit point '''
        points = np.array([self.beam_pos])
        normal_vec = mesh_map.normals_at_points(points ,N, bounds, mesh_reso)[0]
        ''' cal incident angle '''
        project = normal_vec - np.dot(auv_vec, normal_vec)/np.dot(auv_vec, auv_vec) * auv_vec # cal projection of normal in auv_vec plane
        cos_phi = np.dot(beam_vec, project) / np.linalg.norm(beam_vec) / np.linalg.norm(project) # cal angle between beam vector and normal vector
        phi = acos(cos_phi)
        direction = np.dot(auv_vec, np.cross(beam_vec, normal_vec))# decide +-
        phi = phi*direction/abs(direction)

        return phi, angle, beam_dist, beam_vec[2]


''' Get mesh and norm '''
mesh_reso = 0.5 # get mesh from std pings
mesh_file = "/home/chs/Desktop/Sonar/Data/EM2040/mesh.npz"
mesh_data = np.load(mesh_file)
V, F, bounds = mesh_data["V"], mesh_data["F"], mesh_data["bounds"]
N = mesh_map.compute_normals(V, F) # compute normals for the entire mesh(unit vector of reflection surface)


''' Make 4D matrix '''
beam_pattern = {}

# Set para bounds and resolution
deltaz_reso = 1
dist_reso= 1
beamangle_reso = 0.01
incident_reso = 0.01

meas_list = [1]#, 2, 3, 4, 5, 6, 7, 8]
deltaz = []
dist = []
beam_angle = []
incident = []
intensity = []
start = time.time()
for index in meas_list:
    meas_path = '/home/chs/Desktop/Sonar/Data/drape_result/low/meas_data_25.cereal' #% index
    meas_imgs = map_draper.sss_meas_data.read_single(meas_path)
    row, col = np.shape(meas_imgs.sss_waterfall_image) # row and column of waterfall image
    for i in range(row):
        for j in range(col):
            if meas_imgs.sss_waterfall_hits_Z[i,j] == 0:
                continue
            
            hit = beam(i, j, meas_imgs, bounds, N, mesh_reso)
            # cal index in 4D
            deltaz_index     = round(hit.deltaz / deltaz_reso) * deltaz_reso
            dist_index       = round(hit.dist / dist_reso) * dist_reso
            beam_angle_index = round(hit.beam_angle / beamangle_reso ) * beamangle_reso
            incident_index   = round(hit.incident_angle / incident_reso) * incident_reso

            deltaz.append(deltaz_index)
            dist.append(dist_index)
            beam_angle.append(beam_angle_index)
            incident.append(incident_index)
            intensity.append(hit.intensity)
            ''' show beam pattern guess '''
            # if beam_pattern.has_key(beam_angle_index):
            #     v = beam_pattern[beam_angle_index] + hit.intensity * abs(tan(hit.incident_angle))
            #     beam_pattern[beam_angle_index] = (v/2)
            # else:
            #     beam_pattern[beam_angle_index] = hit.intensity * abs(tan(hit.incident_angle))
        if i %100 == 0:
            print("meas_img%d row %d/%d done" % (index, i, row))
end = time.time()
time0 = end-start
print('step1 done, use time %f' % time0)
data = np.array([deltaz, dist, beam_angle, incident, intensity]).T
print(len(data))




''' show beam pattern ''' 
# ax = plt.subplot(111, projection='polar')
# angle = []
# beam = []
# for k1,v1 in beam_pattern.items():
#     k = int(k1) * 0.01
#     angle.append(k)
#     beam.append(v1)
#     print(k, v1)
# ax.scatter(angle, beam, s=2, c='b')
# plt.show()

# ''' Outlier discard and average ''' 
# index = 0
# while index < len(data):
#     row = data[index]
#     if index % 1000 ==0:
#         print("row %d/%d done" % (index, len(data)))
#     key = np.where((data[:,0:4]==row[0:4]).all(1))[0]
#     value = data[key,4]
#     if len(key)==1:
#         index += 1
#         continue
#     # filtered average
#     mean = np.mean(value)
#     std = np.std(value, ddof = 1)
#     elem_list = filter(lambda x: abs((x-mean)/std) < 1.5, value)
#     data[index, 4] = np.mean(elem_list)
#     data = np.delete(data, key[1:],axis=0)
#     index += 1
    
# print('step2 done')
# np.savetxt('/home/chs/Desktop/Sonar/Data/drape_result/Result_second.csv', data, delimiter=',')
# print("???")