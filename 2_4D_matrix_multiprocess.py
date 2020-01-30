#!/usr/bin/env python

from auvlib.data_tools import std_data
from auvlib.bathy_maps import mesh_map, map_draper
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, acos, asin, pi, tan
from multiprocessing import Pool, Manager
import time


def cal_4D_task(hit,intensity,rpy,vehicle_pos,normal_vec,mesh_reso):
    result = []
    Id = []
    for i in range(len(intensity)):
        if hit[i][2] == 0:
            continue
        beam_pos = hit[i]
        roll, pitch, yaw  = rpy[0], rpy[1], rpy[2]
        auv_vec = np.array([cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]) # unit vector of AUV pose
        beam_vec = vehicle_pos - beam_pos # vector from hit point to AUV
        beam_dist = np.linalg.norm(beam_vec)
        ''' cal beam angle '''
        angle = asin(beam_vec[2]/beam_dist)# angle to level,abs
        direction = np.cross(beam_vec[0:2], auv_vec[0:2]) # decide +-
        angle  = angle *direction/abs(direction)
        angle -= roll # fix with roll
        ''' cal incident angle '''
        project = normal_vec[i] - np.dot(auv_vec, normal_vec[i])/np.dot(auv_vec, auv_vec) * auv_vec # cal projection of normal in auv_vec plane
        cos_phi = np.dot(beam_vec, project) / np.linalg.norm(beam_vec) / np.linalg.norm(project) # cal angle between beam vector and normal vector
        phi = acos(cos_phi)
        direction = np.dot(auv_vec, np.cross(beam_vec, normal_vec[i]))# decide +-
        phi = phi*direction/abs(direction)

        # cal index in 4D
        deltaz_index     = round(beam_vec[2] / deltaz_reso) * deltaz_reso
        dist_index       = round(beam_dist / dist_reso) * dist_reso
        beam_angle_index = round(angle / beamangle_reso ) * beamangle_reso
        incident_index   = round(phi / incident_reso) * incident_reso
        result.append([deltaz_index, dist_index, beam_angle_index, incident_index, intensity[i]])
        Id.append(str([deltaz_index, dist_index, beam_angle_index, incident_index])) # Id for sorting with np.unique
    return (result, Id)


''' Add result from multaprocess task '''
data = []
Id_unique = []
def add_list(call_back_result):
    (result, Id) = call_back_result
    for i in range(len(result)):
        data.append(result[i])
        Id_unique.append(Id[i])


''' Import mesh and compute norm '''
mesh_reso = 0.5 # get mesh from std pings
mesh_file = "/home/chs/Desktop/Sonar/Data/EM2040/mesh.npz"
mesh_data = np.load(mesh_file)
V, F, bounds = mesh_data["V"], mesh_data["F"], mesh_data["bounds"]
N = mesh_map.compute_normals(V, F) # compute normals for the entire mesh(unit vector of reflection surface)

''' Make 4D matrix '''
# Set para bounds and resolution
deltaz_reso = 1
dist_reso= 1
beamangle_reso = 0.01
incident_reso = 0.01

# choose layer
LAYER = 1 # 0:high 1:mid 2:low
name = ['high','mid','low']
meas_list = [[33,13,9,30,43,44,24,14,20],[28,36,42,26,16,41,6,38,45],[25,11,21,29,1,19,18,5,17]]

# Init multiprocess list and pool
p = Pool(8)
start = time.time()
for index in meas_list[LAYER]:
    meas_path = '/home/chs/Desktop/Sonar/Data/drape_result/'+name[LAYER]+'/meas_data_%d.cereal' % index
    meas_imgs = map_draper.sss_meas_data.read_single(meas_path)
    row, col = np.shape(meas_imgs.sss_waterfall_image) # row and column of waterfall image
    for i in range(row):
        # get hit points, pos, rpy...(preproduce for mutiprocessing)
        hit = np.array([meas_imgs.sss_waterfall_hits_X[i], #[[x,y,z],[x,y,z]...]
                        meas_imgs.sss_waterfall_hits_Y[i], 
                        meas_imgs.sss_waterfall_hits_Z[i]]).T 
        intensity = meas_imgs.sss_waterfall_image[i] # all waterfall pixel in a row
        rpy = meas_imgs.rpy[i] # roll pitch, yaw of 
        vehicle_pos = meas_imgs.pos[i]
        #normal_vec = mesh_map.normals_at_points(hit ,N, bounds, mesh_reso) # reduce the speed of multitask, use the new one
        x = (hit[:,0]/mesh_reso).astype(int)
        y = (hit[:,1]/mesh_reso).astype(int)
        cols = int((bounds[1,0] - bounds[0,0])/mesh_reso)
        normal_vec = N[y*cols+x]
        # Add a task to the pool
        p.apply_async(cal_4D_task, args=(hit,intensity,rpy,vehicle_pos,normal_vec,mesh_reso),callback = add_list) #.get() 
        # with get.(), error in multiprocess can be reported, but all cores will wait until this process done, used only for debug
        if i %100 == 0:
            print("meas_img%d row %d/%d done" % (index, i, row))
p.close()
p.join()
end = time.time()
time0 = end-start
data = np.array(data)
Id_unique = np.array(Id_unique)
print('step1 done, use time %f' % time0)
print("num of points %d" % len(data))

# for Debug
# np.savetxt('/home/chs/Desktop/Sonar/Data/drape_result/data_'+name[LAYER]+'.csv', data, delimiter=',')
# np.save('/home/chs/Desktop/Sonar/Data/drape_result/id.npy', Id_unique)
# data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/data_low.csv"), delimiter=",")
# Id_unique = np.load('/home/chs/Desktop/Sonar/Data/drape_result/id.npy')

def cal_repeat_task(inverse_index, lst, pool_index):
    print("start task %d" % pool_index)
    for i in lst:
        if i %1000 == 0:
            print("average %d/%d done in pool %d" % (i-pool_index*len(lst), len(lst), pool_index))
        if count[i]!=1:
            key = np.where(inverse_index==i)[0]
            value = data[key,4]
            data[index[i], 4] = np.mean(value)
#     # filtered average
#     mean = np.mean(value)
#     std = np.std(value, ddof = 1)
#     elem_list = filter(lambda x: abs((x-mean)/std) < 1.5, value)
#     data[index, 4] = np.mean(elem_list)


''' Outlieinr discard and average ''' 
unique, index, inverse_index, count = np.unique(Id_unique, return_index=True, return_inverse=True, return_counts=True) 
repet_index = np.delete(np.arange(len(data)), index)
print("len of unique %d" % max(inverse_index))
# start multiprocess to discard outlier and average
p = Pool(8)
start = time.time()
split_index = np.array_split(np.arange(max(inverse_index)), 8)
pool_index = 0
for lst in split_index:
    p.apply_async(cal_repeat_task, args=(inverse_index, lst, pool_index))
    pool_index += 1
p.close()
p.join()
data = np.delete(data, repet_index, axis=0)
end = time.time()
time0 = end-start
print('step2 done, use time %f' % time0)    
np.savetxt('/home/chs/Desktop/Sonar/Data/drape_result/Result_'+name[LAYER]+'.csv', data, delimiter=',')
print("???")