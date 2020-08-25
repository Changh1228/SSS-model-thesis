from auvlib.data_tools import xtf_data
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.ndimage

#Params
xtf_file = "Deconvolution/data/SSH-0013-l02s01-20190618-174142.XTF" #xtf data
height = 20
plot_mean_intensities = True
plot_nadir_edge = True
mode = "cotan" #Available modes are: "cos", "cos2", "cotan"

#-------------------------------------------------------------------------------------------------------

def smoothen_outliers(img, threshold):
    return np.where(np.abs(img) < threshold, img, 0)

def normalize(xtf_sss_ping, llimits, rlimits, height, mode = "cos2", plot = True):
    size = len(xtf_sss_ping.port.pings)
    roll = xtf_sss_ping.roll_
    if np.abs(roll) > np.pi/2:
        roll = 0
        xtf_sss_ping.roll_ = 0
    pitch = xtf_sss_ping.pitch_
    if np.abs(pitch) < np.pi/2:
        height = height/np.cos(pitch)
    else:
        xtf_sss_ping.pitch_ = 0
    #Assuming nadir is at 5 degrees.
    nadir_angle = np.radians(5)

    for side in ["port", "stbd"]:
        new_pings = [None]*size
        if side == "port":
            tmp = xtf_sss_ping.port.pings[:size - llimits]
            tmp = np.where(np.abs(tmp) < np.power(10,6), tmp, 0).tolist()
            pings = xtf_sss_ping.port.pings[size - llimits:]
            new_pings[:size - llimits] = tmp
            slant_range = xtf_sss_ping.port.slant_range
            r_0 = height/np.cos(nadir_angle + roll)
            #Resolution xtf
            delta_r =(slant_range - r_0)/len(pings)
        else:
            tmp = xtf_sss_ping.stbd.pings[:rlimits - size]
            tmp = np.where(np.abs(tmp) < np.power(10,6), tmp, 0).tolist()
            pings = xtf_sss_ping.stbd.pings[rlimits - size:]
            new_pings[:rlimits - size] = tmp
            slant_range = xtf_sss_ping.stbd.slant_range
            r_0 = height/np.cos(nadir_angle - roll)
            #Resolution xtf
            delta_r = (slant_range - r_0)/len(pings)

        ids = np.arange(len(pings))
        phi = np.arccos(height/(r_0 + ids*delta_r))

        #Cotan as normalizing factor
        if mode == "cotan":
            tmp = pings*np.tan(phi)
            tmp = tmp.astype(int)
            tmp = np.where(np.abs(tmp) < np.power(10,6), tmp, 0).tolist()
            new_pings[size - len(pings):] = tmp

        # #Cosine as normalizing factor
        elif mode == "cos":
            tmp = pings/np.cos(phi)
            tmp = tmp.astype(int)
            tmp = np.where(np.abs(tmp) < np.power(10,6), tmp, 0).tolist()
            new_pings[size - len(pings):] = tmp

        #Cosine-squared as normalizing factor.
        elif mode == "cos2":
            tmp = pings/np.power(np.cos(phi), 2)
            tmp = tmp.astype(int)
            tmp = np.where(np.abs(tmp) < np.power(10,6), tmp, 0).tolist()
            new_pings[size - len(pings):] = tmp
        else:
            print("Mode not supported")
            exit()

        new_pings = np.array(new_pings).astype(int)
        new_pings = np.where(np.abs(new_pings) < np.power(10,6), new_pings, 0).tolist()

        #Visualize correction for individual ping
        if plot:
            plt.figure()
            plt.plot(pings)
            plt.figure()
            plt.plot(new_pings)
            plt.show()
        if side == "port":
            xtf_sss_ping.port.pings = new_pings
        else:
            xtf_sss_ping.stbd.pings = new_pings
    return xtf_sss_ping


def _find_nadir(sub_waterfall):
    width = np.size(sub_waterfall,1)
    mid_idx = width/2
    left_limits = []
    right_limits = []

    is_nadir = True
    sides = ["left","right"]

    left_img = cv2.normalize(src=sub_waterfall[:,:mid_idx - mid_idx/5], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    left_img = cv2.equalizeHist(left_img)
    left_img = cv2.medianBlur(left_img, 5)
    left_edges = cv2.Canny(left_img, 100, 220)

    right_img = cv2.normalize(src=sub_waterfall[:,mid_idx + mid_idx/5:], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    right_img = cv2.equalizeHist(right_img)
    right_img = cv2.medianBlur(right_img, 5)
    right_edges = cv2.Canny(right_img, 100, 220)

    edges = np.zeros_like(sub_waterfall)
    edges[:, :mid_idx - mid_idx/5] = left_edges
    edges[:, mid_idx + mid_idx/5:] = right_edges

    for i in range(np.size(sub_waterfall,0)):
        for side in sides:
            is_nadir = True
            idx = mid_idx
            while is_nadir:
                if edges[i,idx] == 255:
                    is_nadir = False
                    if side == "left":
                        left_limits.append(idx)
                    else:
                        right_limits.append(idx)

                elif np.abs(mid_idx - idx) >= mid_idx - 1:
                    if side == "left":
                        left_limits.append(0)
                    else:
                        right_limits.append(0)
                    is_nadir = False
                else:
                    if side == "left":
                        idx -= 1
                    else:
                        idx += 1

    left_limits = scipy.ndimage.median_filter(left_limits, size = 100)
    right_limits = scipy.ndimage.median_filter(right_limits, size = 100)

    return left_limits, right_limits

def plot_nadir(waterfall, left_limits, right_limits, save_fig = False, fig_name = "nadir"):
    img = cv2.normalize(src=waterfall, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img = cv2.equalizeHist(img)
    plt.figure(fig_name, figsize=[8,16])
    plt.imshow(img, cmap = 'gray')
    plt.plot(left_limits, np.arange(np.size(waterfall,0)), 'r')
    plt.plot(right_limits, np.arange(np.size(waterfall,0)), 'r')

    if save_fig == True:
        plt.savefig(fig_name)
    else:
        plt.show()

#Read xtf data
xtf_pings = xtf_data.xtf_sss_ping.parse_file(xtf_file)

counter = 0
start_indices = []
for ping in xtf_pings:
    if ping.first_in_file_:
        start_indices.append(counter)
    counter += 1

#Add final index that is not a start_idx
start_indices.append(counter)

ll, rl = [None]*counter, [None]*counter

for i in range(len(start_indices) - 1):
    print("Finding nadir for image "  +str(i))
    waterfall = xtf_data.make_waterfall_image(xtf_pings[start_indices[i]:start_indices[i+1]])
    waterfall = smoothen_outliers(waterfall, 10)
    width = np.size(waterfall,1)
    mid_idx = width/2
    left_limits, right_limits = _find_nadir(waterfall)
    ll[start_indices[i]:start_indices[i+1]] = left_limits.tolist()
    rl[start_indices[i]:start_indices[i+1]] = right_limits.tolist()
    if plot_nadir_edge:
        plot_nadir(waterfall, left_limits, right_limits)

new_pings = [None]*counter

for i, ping in enumerate(xtf_pings):
    print("Normalizing ping " + str(i + 1) + " of " + str(len(xtf_pings)))
    new_pings[i] = normalize(ping, ll[i], rl[i], height, mode = mode, plot = False)

if plot_mean_intensities:
    for k in range(len(start_indices) - 1):
        I_port = np.zeros((start_indices[k+1] - start_indices[k], len((new_pings[0]).port.pings)))
        I_stbd = np.zeros((start_indices[k+1] - start_indices[k], len((new_pings[0]).stbd.pings)))

        for i,j in enumerate(range(start_indices[k], start_indices[k+1])):
            I_port[i,:] = new_pings[j].port.pings
            I_stbd[i,:] = new_pings[j].stbd.pings
        port_mean = np.flip(np.mean(I_port, 0)).tolist()
        stbd_mean = np.mean(I_stbd, 0).tolist()
        mean = port_mean + stbd_mean
        plt.figure("Mean intensity " + str(k) + " " + mode)
        plt.plot(mean)
        plt.show()

#xtf_data.write_data(new_pings, "/home/auv/auv/datasets/20190618_6/processed_data/normalized_xtf/" + mode + "_xtf.cereal")
