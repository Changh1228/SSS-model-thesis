''' Correct sonar image based on discrete correction factors '''
from auvlib.bathy_maps import mesh_map, map_draper
from math import sqrt, cos, sin, acos, asin, pi, tan
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import Image



def correction(meas_imgs, N, bounds, img_name, factor1, factor2, factor3):
    deltaz_reso = 1
    dist_reso= 1
    beamangle_reso = 0.01
    incident_reso = 0.01

    index = ['deltaz-a', 'distance-r', 'beam_angle-theta', 'incident_angle-phi']

    row, col = np.shape(meas_imgs.sss_waterfall_image)
    img1 = np.zeros((row, col))
    img2 = np.zeros((row, col))
    img3 = np.zeros((row, col))
    for i in range(row):
        rpy = meas_imgs.rpy[i] # roll pitch, yaw
        roll, pitch, yaw  = rpy[0], rpy[1], rpy[2]
        vehicle_pos = meas_imgs.pos[i]
        for j in range(col):
            hit = np.array([meas_imgs.sss_waterfall_hits_X[i][j], #[[x,y,z],[x,y,z]...]
                            meas_imgs.sss_waterfall_hits_Y[i][j], 
                            meas_imgs.sss_waterfall_hits_Z[i][j]])
            if hit[2] == 0:
                continue
            intensity = meas_imgs.sss_waterfall_image[i][j] # all waterfall pixel in a row
            x = (hit[0]/mesh_reso).astype(int)
            y = (hit[1]/mesh_reso).astype(int)
            cols = int((bounds[1,0] - bounds[0,0])/mesh_reso)
            normal_vec = N[y*cols+x]
            beam_pos = hit
            auv_vec = np.array([cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]) # unit vector of AUV pose
            beam_vec = vehicle_pos - beam_pos # vector from hit point to AUV
            beam_dist = np.linalg.norm(beam_vec)
            ''' cal beam angle '''
            angle = acos(beam_vec[2]/beam_dist)# angle w.r.t vertical,abs
            direction = np.cross(auv_vec[0:2], beam_vec[0:2]) # decide +-
            angle  = angle *direction/abs(direction)
            angle -= roll # fix with roll
            ''' cal incident angle '''
            project = normal_vec - np.dot(auv_vec, normal_vec)/np.dot(auv_vec, auv_vec) * auv_vec # cal projection of normal in auv_vec plane
            cos_phi = np.dot(beam_vec, project) / np.linalg.norm(beam_vec) / np.linalg.norm(project) # cal angle between beam vector and normal vector
            phi = acos(cos_phi)
            direction = np.dot(auv_vec, np.cross(beam_vec, normal_vec))# decide +-
            phi = phi*direction/abs(direction)
            ''' cal index in 4D '''
            deltaz_index     = round(beam_vec[2] / deltaz_reso) * deltaz_reso
            dist_index       = round(beam_dist / dist_reso) * dist_reso
            beam_angle_index = round(angle / beamangle_reso ) /100#beamangle_reso
            incident_index   = round(phi / incident_reso) /100#incident_reso
            #index = [deltaz_index, dist_index, beam_angle_index, incident_index]
            ''' Correction '''
            # distance
            key = np.where(factor1[:,0]==dist_index)[0][0]
            intensity *= factor1[key][1]
            img1[i][j] = intensity
            # beam angle
            key = np.where(abs(factor2[:,0])==abs(beam_angle_index))[0][0]
            intensity *= np.mean(factor2[key][1])
            img2[i][j] = intensity
            # incidence angle 
            key = np.where(abs(factor3[:,0])==abs(incident_index))[0][0]
            #intensity *= np.mean(factor3[key][1])
            intensity *= abs(tan(incident_index)) #np.mean(factor3[key][1])

            if intensity < 6:
                img3[i][j] = intensity
    # plt.figure(0, figsize=(16,9))
    # plt.subplot(2,2,1)
    # plt.imshow(meas_imgs.sss_waterfall_image)
    # plt.subplot(2,2,2)
    # plt.imshow(img1)
    # plt.subplot(2,2,3)
    # plt.imshow(img2)
    # plt.subplot(2,2,4)
    # plt.imshow(img3)
    # plt.savefig('/home/chs/Desktop/Sonar/Img/'+img_name+'_a')

    # draw average furve
    # for j in range(col):
    #     img1[0,j] = np.mean(filter(lambda x: x!=0, img1[:,j]))
    #     img2[0,j] = np.mean(filter(lambda x: x!=0, img2[:,j]))
    #     img3[0,j] = np.mean(filter(lambda x: x!=0, img3[:,j]))
    # img_ori = np.mean(meas_imgs.sss_waterfall_image, axis = 0)
    # # compute variance
    # var_ori = np.var(img_ori)
    # var_1 = np.var(img1[0])
    # var_2 = np.var(img2[0])
    # var_3 = np.var(img3[0])
    # x = np.arange(col)
    # plt.figure(1, figsize=(16,9))
    # plt.subplot(2,2,1)
    # plt.ylim(0,9)
    # plt.title("ori_img var= %f" % (var_ori/row))
    # plt.scatter(x, img_ori, s=1, c='b')
    # plt.subplot(2,2,2)
    # plt.ylim(0,9)
    # plt.title("correct r var= %f" % (var_1/row))
    # plt.scatter(x, img1[0], s=1, c='b')
    # plt.subplot(2,2,3)
    # plt.ylim(0,9)
    # plt.title("correct theta var= %f" % (var_2/row))
    # plt.scatter(x, img2[0], s=1, c='b')
    # plt.subplot(2,2,4)
    # plt.ylim(0,9)
    # plt.title("correct phi var= %f" % (var_3/row))
    # plt.scatter(x, img3[0], s=1, c='b')
    # plt.savefig('/home/chs/Desktop/Sonar/Img/'+img_name+'_b')

    #plt.figure(2, figsize=(16,9))
    #plt.imshow(img3)
    #plt.savefig('/home/chs/Desktop/Sonar/Img/'+img_name+'_c')
    #pltimg.imsave('/home/chs/Desktop/Sonar/Img/'+img_name+'_c.png', img3)
    print(img3.min(), img3.max())
    I8 = (((img3 - img3.min())/(img3.max() - img3.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(I8)
    img.save('/home/chs/Desktop/Sonar/Img/'+img_name+'_c.png')

    plt.clf()
    return



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
later_name = ['high','mid','low']
meas_list = [[33,13,9,30,43,44,24,14,20],[16],[25,11,21,29,1,19,18,5,17]] # 28,36,42,26,16,41,6,38,45

#factor = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor" + index[dim] +".csv") , delimiter=",")
factor1 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor_distance-r.csv") , delimiter=",")
factor2 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor_beam_angle-theta.csv") , delimiter=",")
factor3 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/factor_incident_angle-phi_filtered.csv") , delimiter=",")

for index in meas_list[LAYER]:
    meas_path = '/home/chs/Desktop/Sonar/Data/drape_result/'+later_name[LAYER]+'/meas_data_%d.cereal' % index
    meas_imgs = map_draper.sss_meas_data.read_single(meas_path)
    img_name = 'corrected_'+str(LAYER)+'_'+str(index)
    correction(meas_imgs, N, bounds, img_name, factor1, factor2, factor3)

print("Orz")