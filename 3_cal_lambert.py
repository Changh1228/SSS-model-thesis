#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import math


def intensity_ave_task(inverse_index, lst, pool_index, data, count):
    print("start task %d" % pool_index)
    result = np.zeros(6)
    for i in lst:
        if i %1000 == 0:
            print("average %d/%d done in pool %d" % (i-pool_index*len(lst), len(lst), pool_index))
        key = np.where(inverse_index==i)[0]
        if count[i]!=1:
            value = data[key,4]
            data[key, -1] = np.mean(value)
        else:
            data[key, -1] = data[key, 4]
        result = np.vstack((result, data[key]))
    return result[1:]

''' calibrate deltaz (use gradent dencent)'''
def intensity_ave(data, dim):
    '''[calculate average intensity with same paramater after dim(dim not included)]
    
    :param data: [dataset]
    :type data: [ndarray shape(n, 6)]
    :param dim: [dimension of parameter we want to average]
    :type dim: [int, range: 0~3]
    :return: [dataset after average]
    :rtype: [ndarray]
    '''
    if dim != 3:
        idx = np.array([str(data[i, dim+1:4]) for i in range(len(data))])
        unique, index, inverse_index, count = np.unique(idx, return_index=True, return_inverse=True, return_counts=True)
        print("len of unique %d" % max(inverse_index))
        split_index = np.array_split(np.arange(max(inverse_index)), 8)
        p = Pool(8)
        start = time.time()
        pool_index = 0
        result = []
        for lst in split_index:
            result.append(p.apply_async(intensity_ave_task, args=(inverse_index, lst, pool_index, data, count)))
            pool_index += 1
        p.close()
        p.join()
        data = np.zeros(6)
        for item in result:
            data = np.vstack((data, item.get())) 
        data = data[1:]
        end = time.time()
        time0 = end-start
        print('Average in dim %d done, use time %f' % (dim, time0))
    else: # dim = 3 is the last dimension, not necessary to find same para
        value = data[:,4]
        mean = np.mean(value)
        data[:,-1] = mean
        print("Average in dim %d done" % dim)
    return data


def calibrate_task(data, inverse_index, lst, correction, dim, pool_index):
    result = []
    for i in lst:
        if i %1000 == 0:
            print("Calibration %d/%d done in pool %d" % (i-pool_index*len(lst), len(lst), pool_index))
        keyi = np.where(inverse_index==i)[0]
        factor = np.zeros(len(keyi))
        for j in range(len(keyi)): # find correspinding calibrate factor
            keyj = np.where(correction==data[keyi[j],dim])[0]
            factor[j] =  correction[keyj[0], 1]
        value = data[keyi,-2]
        y = factor*value
        data[keyi[0], 4] = np.mean(y)
        result.append(data[keyi[0]])
    return result


def calibrate(data, dim, correction): 
    '''[function correcting para[dim] using polynomial or discrete function ]
    
    :param data: [dataset]
    :type data: [ndarray shape(n,6) (para before dim is useless)]
    :param dim: [dimension of parameter for correction]
    :type dim: [int range 0~2 (dim3 don't need correction)]
    :param correction: [polynominal or discrete correct function]
    :type correction: [poly1d or ndarray(2, n)]
    :return: [dataset after correction]
    :rtype: [ndarray shape(n,6)]
    '''
    # compute calibration result
    idx = np.array([str(data[i, dim+1:4]) for i in range(len(data))])
    unique, index, inverse_index = np.unique(idx, return_index=True, return_inverse=True)
    print("num of corrected data %d" % max(inverse_index))
    split_index = np.array_split(np.arange(max(inverse_index)), 8)
    p = Pool(8)
    start = time.time()
    pool_index = 0
    result = []
    #calibrate_task(inverse_index, split_index[0], correction, dim, pool_index)
    for lst in split_index:
        result.append(p.apply_async(calibrate_task, args=(data, inverse_index, lst, correction, dim, pool_index)))
        pool_index += 1
    p.close()
    p.join()
    end = time.time()
    time0 = end-start
    data = []
    for item in result:
        data += item.get()
    data = np.array(data)
    y = data[:, 4]
    y_ave = data[:, 5]
    catch = y - y_ave
    squared_err = catch*catch
    res = np.sqrt(np.mean(squared_err))
    print('RMSE after calibration %s'% abs(res))
    print('Calibration in dim %d done, use time %f' % (dim, time0))
    return data


def discrete_correction(data, dim):
    '''[calibrate parameter in dim with dicrete function]
    
    :param data: [dataset]
    :type data: [ndarray shape(n,6) (para before dim is useless)]
    :param dim: [dimension of parameter for correction]
    :type dim: [int range 0~3]
    :return: [dataset after correction]
    :rtype: [ndarray]
    '''
    ''' compute discrete correction function'''
    data = intensity_ave(data, dim) # cal average of intensity with same para after dim and different para in dim
    unique, index, inverse_index= np.unique(data[:,dim], return_index=True, return_inverse=True)
    print("num of factor %d" % max(inverse_index))
    correction = []# discrete correction value [para, correction]
    for i in range(len(unique)): # look for data with same para[dim]
        key = np.where(inverse_index==i)[0]
        j = data[key,4]
        j_bar = data[key,5]
        c_dim  = np.sum(j*j_bar)/np.sum(j**2)
        correction.append([unique[i], c_dim])
    # print(correction)
    ''' Plot correction function '''
    index = ['incident_angle-phi', 'distance-r', 'beam_angle-theta', 'deltaz-a']
    plt.figure(0)
    plt.grid()
    plt.title(index[dim])
    correction = np.array(correction)
    np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/factor"+index[dim]+".csv", correction, delimiter=',')
    if dim == 0:
        plt.ylim(-2,10)
        plt.scatter(correction[:,0], correction[:,1], s=2, c='b', marker='.')
        #plt.plot(correction[:,0], correction[:,1])
        plt.plot(correction[:,0], abs(np.tan(correction[:,0])), c='r', label = '1/cot')
        plt.plot(correction[:,0], abs(1/np.cos(correction[:,0])), c='g', label = '1/cos')
        plt.plot(correction[:,0], 1/np.cos(correction[:,0])**2, c='y', label = '1/cos^2')
        plt.legend()
    else:
        plt.scatter(correction[:,0], correction[:,1], s=2, c='b', marker='.')

    plt.show()
    ''' Calibrate intensity for next dim '''
    if dim != 3:
        data = calibrate(data, dim, correction)
    else: # last dimension don't need correction
        pass
    return data
    
''' calibrate incident_angle-phi ''' 
# data1 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_high.csv"), delimiter=",") # load data
# data2 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_mid.csv") , delimiter=",")
# data3 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_low.csv") , delimiter=",")
# data = np.vstack((data2, data3))
# data = np.hstack((data, np.zeros((len(data),1)))) # add a column for lower dimention average
# data = discrete_correction(data, 3)
# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_0deltaz.csv", data, delimiter=',')

''' calibrate distance (use discrete correct function)'''
# data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_0deltaz.csv"), delimiter=",")
# data = discrete_correction(data, 1)
# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_1distance.csv", data, delimiter=',')

''' calibrate beam angle '''
# data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_1distance.csv"), delimiter=",")
# data = discrete_correction(data, 2)
# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_2beam_angle.csv", data, delimiter=',')

''' calibrate incidence angle '''
# data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_test.csv"), delimiter=",")
# data = discrete_correction(data, 3)
# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_3incidence_angle.csv", data, delimiter=',')

print("???")
