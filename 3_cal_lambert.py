#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
#from multiprocessing import Pool


''' calibrate deltaz (use gradent dencent)'''
def intensity_ave(data, dim): # TODO: multiprocess
    '''[calculate average intensity with same paramater after dim(dim not included)]
    
    :param data: [dataset]
    :type data: [ndarray shape(n, 6)]
    :param dim: [dimension of parameter we want to average]
    :type dim: [int, range: 0~3]
    :return: [dataset after average]
    :rtype: [ndarray]
    '''
    if dim != 3:
        for index, row in enumerate(data):
            if index %1000 ==0:
                print("Ave row %d/%d done" % (index, len(data)))
            key = np.where((data[:,dim+1:4]==row[dim+1:4]).all(1))[0]
            value = data[key,4]
            mean = np.mean(value)
            data[key, -1] = mean
    else: # dim = 3 is the last dimension, not necessary to find same para
        value = data[:,4]
        mean = np.means(value)
        data[:,-1] = mean
    print("Average in dim %d done" % dim)
    return data

def cal_grad(a, y, y_ave, beta):
    order = len(beta.c)
    grad = np.zeros(order)
    a1 = beta(a)
    a2 = beta(a)*y
    a3 = beta(a)*y - y_ave
    cache = y*(beta(a)*y - y_ave)
    i = order -1
    while i>-1:
        grad[i] = np.mean(cache)
        cache *= a
        i -= 1
    # grad[0] = np.mean(a*a*catch)
    # grad[1] = np.mean(a*catch)
    # grad[2] = np.mean(catch)
    return grad

def update_beta(beta, alpha, grad):
    new_beta = beta - alpha*grad
    return new_beta

def rmse(a, y, y_ave, beta):
    catch = beta(a)*y - y_ave
    squared_err = catch*catch
    res = np.sqrt(np.mean(squared_err))
    return res

def grad_desc(data, dim, beta):
    # dim: choosen gradient decent dimension
    print('Gradiet Decent of %dth dimenstion start'%dim)
    # Init
    alpha = 0.2 # learing rate
    tol_l = 0.01
    # Normalize data
    max_x = data[:,dim].max() 
    data[:,dim] /= max_x
    data = intensity_ave(data, dim)

    a = data[:, dim]
    y = data[:, -2]
    y_ave = data[:, -1]
    # first round
    grad = cal_grad(a, y, y_ave, beta)
    loss = rmse(a, y, y_ave, beta)
    beta = update_beta(beta, alpha, grad)
    loss_new = rmse(a, y, y_ave, beta)

    i = 1
    while i < 1000:
        grad = cal_grad(a, y, y_ave, beta)
        loss = rmse(a, y, y_ave, beta)
        beta = update_beta(beta, alpha, grad)
        loss_new = rmse(a, y, y_ave, beta)
        if i % 10 ==0:
            print('Round %s Diff RMSE %s, ABS RMSE %s'%(i, abs(loss_new - loss), loss_new), beta)
        i += 1
    # unscale the model function because normalize
    order = len(beta.c)
    beta_real = np.zeros(order)
    i = order-1
    cache = 1
    while i > -1: # beta_real = beta.c/[max**2, max, 1]
        beta_real[i] = beta.c[i]/cache
        cache *= max_x
        i -= 1
    data[:,dim] *= max_x
    beta_real = np.poly1d(beta_real)
    print('Done with RMSE %s and beta'% abs(loss_new), beta_real)

    # plot model
    index = ['deltaz-a', 'distance-r', 'beam_angle-theta', 'incident_angle-phi']
    a = data[:,dim]
    y  = beta_real(a)
    plt.figure(0)
    plt.grid()
    plt.title(index[dim])
    plt.scatter(a, y, s=2, c='b', marker='.')
    plt.show()
    return beta_real, data

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
    # compute calibration result TODO: multiprocess
    idx = np.array([str(data[i, dim+1:4]) for i in range(len(data))])
    unique, index, inverse_index, count = np.unique(idx, return_index=True, return_inverse=True, return_counts=True) 
    repet_index = np.delete(np.arange(len(data)), index)
    if type(correct) is np.poly1d: # poly correction
        for i in range(len(unique)):
            key = np.where(inverse_index==i)[0]
            value = data[key,-2]
            a = data[key,dim]
            y = correction(a)*value
            data[index[i],-2] = np.mean(y)
    else: # discrete correction
        # find correspinding calibrate value
        factor = np.zeros(len(data))
        for i in range(len(data)):
            key = np.where(correction==data[i,dim])[0]
            if len(key)>1: # debug
                print("???")
            factor[i] =  correction[key[0], 1]
        for i in range(len(unique)):
            key = np.where(inverse_index==i)[0]
            value = data[key,-2]
            a = data[key,dim]
            y = factor[key]*a
            data[index[i],-2] = np.mean(y)
    data = np.delete(data, repet_index, axis=0)
    y = data[:, 4]
    y_ave = data[:, 5]
    catch = y - y_ave
    squared_err = catch*catch
    res = np.sqrt(np.mean(squared_err))
    print('RMSE after calibration %s'% abs(res))
    return data


data1 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_high.csv"), delimiter=",") # load data
data2 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_mid.csv") , delimiter=",")
data3 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_low.csv") , delimiter=",")
data = np.vstack((data1, data2, data3))
data = np.hstack((data, np.zeros((len(data),1)))) # add a column for lower dimention average
beta = np.poly1d([1., 1., 1.]) # choose order of model(c(a) = b0a^2 + b1a +b2)
beta, data = grad_desc(data, 0, beta) # cal model for deltaz
data = calibrate(data, 0, beta) # calibrate deltaz
np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_0deltaz.csv", data, delimiter=',')


''' calibrate distance (use discrete correct function)'''
def discrete_calibrate(data, dim): # TODO: multiprocess
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
    correction = []# discrete correction value [para, correction]
    for i in range(len(unique)): # look for data with same para[dim]
        key = np.where(inverse_index==i)[0]
        j = data[key,4]
        j_bar = data[key,5]
        c_dim  = np.sum(j*j_bar)/np.sum(j_bar**2)
        correction.append([unique[i], c_dim])
    
    ''' Plot correction function '''
    index = ['deltaz-a', 'distance-r', 'beam_angle-theta', 'incident_angle-phi']
    plt.figure(0)
    plt.grid()
    plt.title(index[dim])
    plt.scatter(correction[:,0], correction[:1], s=2, c='b', marker='.')
    plt.show()
    ''' Calibrate intensity '''
    if dim != 3:
        data = calibrate(data, dim, correction)
    else: # last dimension don't need correction
        pass
    return data
    

data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_0deltaz.csv"), delimiter=",")
data = discrete_calibrate(data, 1) # calibrate distance
np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_1distance.csv", data, delimiter=',')

''' calibrate beam angle '''
data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_1distance.csv"), delimiter=",")
data = discrete_calibrate(data, 2) # calibrate distance
np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_2beam_angle.csv", data, delimiter=',')

''' calibrate incidence angle '''
data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_2beam_angle.csv"), delimiter=",")
data = discrete_calibrate(data, 3) # calibrate distance
np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_3incidence_angle.csv", data, delimiter=',')
print("???")