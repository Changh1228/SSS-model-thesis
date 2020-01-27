#!/usr/bin/env python

from auvlib.data_tools import std_data
from auvlib.bathy_maps import mesh_map, map_draper
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, acos, asin, pi, tan
from multiprocessing import Pool, Manager
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
        cols = int((bounds[1,0] - bounds[0,0])/mesh_reso)
        x = int(self.beam_pos[0]/mesh_reso) 
        y = int(self.beam_pos[1]/mesh_reso)
        normal_vec = N[y*cols+x]# direction vector of reflection plane
        ''' cal incident angle '''
        project = normal_vec - np.dot(auv_vec, normal_vec)/np.dot(auv_vec, auv_vec) * auv_vec # cal projection of normal in auv_vec plane
        cos_phi = np.dot(beam_vec, project) / np.linalg.norm(beam_vec) / np.linalg.norm(project) # cal angle between beam vector and normal vector
        phi = acos(cos_phi)
        direction = np.dot(auv_vec, np.cross(beam_vec, normal_vec))# decide +-
        phi = phi*direction/abs(direction)

        return phi, angle, beam_dist, beam_vec[2]


def add_list(hit,intensity,rpy,vechile_pos,normal_vec,bounds,mesh_reso,deltaz_ls,dist_ls,beam_ls,incident_ls,intensity_ls):
    beam_pos = np.array(hit)
    
    # bounds [[xmin, ymin],[xmax, ymax]] of mesh; N normal of whole mesh
    roll  = rpy[0]
    pitch = rpy[1]
    yaw   = rpy[2]
    auv_vec = np.array([cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]) # unit vector of AUV pose
    beam_vec = vechile_pos - beam_pos # vector from hit point to AUV
    beam_dist = np.linalg.norm(beam_vec)
    ''' cal beam angle '''
    angle = asin(beam_vec[2]/beam_dist)# angle to level,abs
    direction = np.cross(beam_vec[0:2], auv_vec[0:2]) # decide +-
    angle  = angle *direction/abs(direction)
    angle -= roll # fix with roll
    ''' get normal vector at hit point '''
    # cols = int((bounds[1,0] - bounds[0,0])/mesh_reso)
    # x = int(beam_pos[0]/mesh_reso) 
    # y = int(beam_pos[1]/mesh_reso)
    # normal_vec = N[y*cols+x]# direction vector of reflection plane
    ''' cal incident angle '''
    project = normal_vec - np.dot(auv_vec, normal_vec)/np.dot(auv_vec, auv_vec) * auv_vec # cal projection of normal in auv_vec plane
    cos_phi = np.dot(beam_vec, project) / np.linalg.norm(beam_vec) / np.linalg.norm(project) # cal angle between beam vector and normal vector
    phi = acos(cos_phi)
    direction = np.dot(auv_vec, np.cross(beam_vec, normal_vec))# decide +-
    phi = phi*direction/abs(direction)

    # cal index in 4D
    deltaz_index     = round(beam_vec[2] / deltaz_reso) * deltaz_reso
    dist_index       = round(beam_dist / dist_reso) * dist_reso
    beam_angle_index = round(angle / beamangle_reso ) * beamangle_reso
    incident_index   = round(phi / incident_reso) * incident_reso
    


''' Import multibeam data '''
std_file = "/home/chs/Desktop/Sonar/Data/EM2040/mid/pings_centered.cereal" 
std_pings =  std_data.mbes_ping.read_data(std_file)

''' Get mesh and norm '''
mesh_reso = 0.5 # get mesh from std pings
V, F, bounds = mesh_map.mesh_from_pings(std_pings, mesh_reso)
N = mesh_map.compute_normals(V, F) # compute normals for the entire mesh(unit vector of reflection surface)

''' Make 4D matrix '''

# Set para bounds and resolution
deltaz_reso = 1
dist_reso= 1
beamangle_reso = 0.01
incident_reso = 0.01

meas_list = [1]#, 2, 3, 4, 5, 6, 7, 8] # measure image index list

# Init multiprocess list and pool
deltaz_ls = Manager().list([])
dist_ls = Manager().list([])
beam_ls = Manager().list([])
incident_ls = Manager().list([])
intensity_ls = Manager().list([])
p = Pool(8)
start = time.time()
for index in meas_list:
    meas_path = '/home/chs/Desktop/Sonar/Data/drape_result/mid/meas_data_%d.cereal' % index
    meas_imgs = map_draper.sss_meas_data.read_single(meas_path)
    row, col = np.shape(meas_imgs.sss_waterfall_image) # row and column of waterfall image

    for i in range(row):
        for j in range(col):
            if meas_imgs.sss_waterfall_hits_Z[i,j] == 0:
                continue
            # start = time.time()
            # get hit points, pos, rpy...(preproduce for mutiprocessing)
            hit = [meas_imgs.sss_waterfall_hits_X[i, j], 
                   meas_imgs.sss_waterfall_hits_Y[i, j], 
                   meas_imgs.sss_waterfall_hits_Z[i, j]]
            intensity = meas_imgs.sss_waterfall_image[i,j]
            rpy = meas_imgs.rpy[i]
            vechile_pos = meas_imgs.pos[i]
            x = int(hit[0]/mesh_reso) 
            y = int(hit[1]/mesh_reso)
            cols = int((bounds[1,0] - bounds[0,0])/mesh_reso)
            normal_vec = N[y*cols+x]
            p.apply_async(add_list, args=(hit,intensity,rpy,vechile_pos,normal_vec,bounds,mesh_reso,
                                          deltaz_ls,dist_ls,beam_ls,incident_ls,intensity_ls))
            # end = time.time()
            # time0 = end-start
            # print('step0 done, use time %f' % time0)
        if i == 100:
            print("step2 done")
            break
        if i %100 == 0:
            print("meas_img%d row %d/%d done" % (index, i, row))
        
        
p.close()
p.join()
end = time.time()
time0 = end-start
print('step1 done, use time %f' % time0)
data = np.array([deltaz_ls,dist_ls,beam_ls,incident_ls,intensity_ls]).T



''' Outlier discard and average ''' 
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
#np.savetxt('/home/chs/Desktop/Sonar/Data/drape_result/Result_second.csv', data, delimiter=',')
# print("???")