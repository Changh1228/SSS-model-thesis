#!/usr/bin/env python

from auvlib.data_tools import std_data
from auvlib.bathy_maps import mesh_map, map_draper
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, acos, asin, pi, tan
from multiprocessing import Pool
import time
import sys
from tf.transformations import euler_matrix

EPS = sys.float_info.epsilon


def cal_4D_task(hit,intensity,rpy,vehicle_pos,normal_vec,mesh_reso):
    result = []
    Id = []
    for i in range(len(intensity)):
        if hit[i][2] == 0:
            continue
        beam_pos = hit[i]
        roll, pitch, yaw  = rpy[0], rpy[1], rpy[2] # positive: clockwise
        auv_vec = np.array([cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]) # unit vector of AUV pose
        beam_vec = vehicle_pos - beam_pos # vector from hit point to AUV
        ''' correct consor offset '''
        direction = np.cross(beam_vec[0:2], auv_vec[0:2]) # port(left)+, stbd(right)-
        r = 0.4375
        d_y = r*cos(pi/4) * direction / abs(direction) # add difference for port and stbd
        d_z = r*(1-cos(pi/4))
        r_matrix = euler_matrix(0, pitch, yaw) # TODO: ignore roll?
        offset = np.dot(r_matrix[:3,:3], np.array([0, d_y, d_z]))
        sensor_pos = vehicle_pos + offset
        # update basic vectors
        beam_vec = sensor_pos - beam_pos # vector from hit point to AUV
        beam_dist = np.linalg.norm(beam_vec)

        ''' cal beam angle '''
        angle = acos(beam_vec[2]/beam_dist)# angle w.r.t vertical,abs
        direction = np.cross(beam_vec[0:2], auv_vec[0:2]) # port(left)+, stbd(right)-
        angle  = angle * direction/abs(direction)
        angle -= roll # fix with roll
        ''' cal incident angle '''
        project = normal_vec[i] - np.dot(auv_vec, normal_vec[i])/np.dot(auv_vec, auv_vec) * auv_vec # cal projection of normal in auv_vec plane
        cos_phi = np.dot(beam_vec, project) / np.linalg.norm(beam_vec) / np.linalg.norm(project) # cal angle between beam vector and normal vector
        phi = acos(cos_phi)
        direction = np.dot(auv_vec, np.cross(normal_vec[i], beam_vec))# decide +-
        phi = phi*direction/abs(direction)

        # cal index in 4D
        deltaz_index     = round(beam_vec[2] / deltaz_reso) * deltaz_reso
        dist_index       = round(beam_dist / dist_reso) * dist_reso
        beam_angle_index = round(angle / beamangle_reso ) * beamangle_reso
        incident_index   = round(phi / incident_reso) * incident_reso
        Id = str([incident_index, dist_index, beam_angle_index, deltaz_index]) # Id for sorting with np.unique
        result.append([deltaz_index, dist_index, beam_angle_index, incident_index, intensity[i], Id])
        # phi, r, theta, a
    return result


''' Add result from multaprocess task '''
data = []
def add_list(result):
    for data_point in result:
        data.append(data_point)


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
LAYER = 2 # 0:high 1:mid 2:low
name = ['high','mid','low']
meas_list = [[33,13,9,30,43,44,24,14,20],[28,36,42,26,16,41,6,38],[19,18,5,17]]# 11,21,29,1,19,18,5,17

# Init multiprocess list and pool
p = Pool(8)
start = time.time()
for index in meas_list[LAYER]:
    meas_path = '/home/chs/Desktop/Sonar/Data/drape_result/'+name[LAYER]+'/meas_data_%d.cereal' % index
    meas_imgs = map_draper.sss_meas_data.read_single(meas_path)
    row, col = np.shape(meas_imgs.sss_waterfall_image) # row and column of waterfall image
    for i in range(row)[100:-100]:
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
        p.apply_async(cal_4D_task, args=(hit,intensity,rpy,vehicle_pos,normal_vec,mesh_reso),callback = add_list)#.get()
        # with get.(), error in multiprocess can be reported, but all cores will wait until this process done, used only for debug
        if i %100 == 0:
            print("meas_img%d row %d/%d done" % (index, i, row))
p.close()
p.join()
end = time.time()
time0 = end-start
Id_unique = np.array([data[i][-1] for i in range(len(data))])
data = np.array([data[i][0:5] for i in range(len(data))])
print('step1 done, use time %f' % time0)
print("num of points %d" % len(data))

def cal_repeat_task(inverse_index, lst, count, pool_index, data):
    print("start task %d" % pool_index)
    result = []
    for i in lst:
        if i %1000 == 0:
            print("average %d/%d done in pool %d" % (i-pool_index*len(lst), len(lst), pool_index))
        key = np.where(inverse_index==i)[0]
        value = data[key,4]
        if count[i]!=1:
            # filtered average
            mean = np.mean(value)
            std = np.std(value, ddof = 1)+EPS
            elem_list = filter(lambda x: abs((x-mean)/std) < 1.3, value)
            data[key[0], 4] = np.mean(elem_list)
        result.append(data[key[0]])
    return result


''' Outlieinr discard and average '''
unique, index, inverse_index, count = np.unique(Id_unique, return_index=True, return_inverse=True, return_counts=True)
print("len of unique %d" % max(inverse_index))
# start multiprocess to discard outlier and average
p = Pool(8)
start = time.time()
split_index = np.array_split(np.arange(max(inverse_index)), 8)
pool_index = 0
result = []
for lst in split_index:
    result.append(p.apply_async(cal_repeat_task, args=(inverse_index, lst, count, pool_index, data)))
    pool_index += 1
p.close()
p.join()
data = []
for item in result:
    data += item.get()
data = np.array(data)
end = time.time()
time0 = end-start
print('step2 done, use time %f' % time0)
np.savetxt('/home/chs/Desktop/Sonar/Data/drape_result/Result_'+name[LAYER]+'.csv', data, delimiter=',')
print("???")
