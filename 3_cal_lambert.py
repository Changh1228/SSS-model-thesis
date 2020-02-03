#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


''' calibrate deltaz (use gradent dencent)'''
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
    # prepair averaged data for loss function
    for index, row in enumerate(data):
        if index %1000 ==0:
            print("Ave row %d/%d done" % (index, len(data)))
        if row[-1] != 0:
            continue
        key = np.where((data[:,dim:4]==row[dim:4]).all(1))[0]
        value = data[key,4]
        mean = np.mean(value)
        data[key, -1] = mean
    print('Step1 done')

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
    index = ['deltaz', 'distance', 'beam_angle', 'incident_angle']
    a = data[:,dim]
    y  = beta_real(a)
    plt.figure(0)
    plt.grid()
    plt.title(index[dim])
    plt.scatter(a, y, s=2, c='b', marker='.')
    plt.show()
    
    return beta_real, data

def calibrate(data, dim, beta): # compute calibration result
    index = 0
    while index < len(data):
        if index %1000 ==0:
            print("row %d/%d done" % (index, len(data)))
        row = data[index]
        key = np.where((data[:,dim+1:4]==row[dim+1:4]).all(1))[0]
        value = data[key,-2]
        a = data[key,dim]
        y = beta(a)*value
        data[index,-2] = np.mean(y)
        data = np.delete(data, key[1:],axis=0)
        index += 1

    a = data[:, dim]
    y = data[:, 4]
    y_ave = data[:, 5]
    catch = y - y_ave
    squared_err = catch*catch
    res = np.sqrt(np.mean(squared_err))
    print('RMSE after calibration %s'% abs(res))
    return data


data1 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_high.csv"), delimiter=",") # load data
data2 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_mid.csv"), delimiter=",")
data3 = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_low.csv"), delimiter=",")
data = np.vstack((data1, data2, data3))
data = np.hstack((data, np.zeros((len(data),1)))) # add a column for lower dimention average
beta = np.poly1d([1., 1., 1.]) # choose order of model(c(a) = b0a^2 + b1a +b2)
beta, data = grad_desc(data, 0, beta) # cal model for deltaz
data = calibrate(data, 0, beta) # calibrate deltaz
np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_0deltaz.csv", data, delimiter=',')


''' calibrate distance (use discrete correct function)'''

# data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_0deltaz.csv"), delimiter=",")
# beta = np.poly1d([1., 1., 1., 1.]) # choose order of model(c(a) = b0a^2 + b1a +b2)
# beta, data = grad_desc(data, 1, beta) # cal model for distance
# data = calibrate(data, 1, beta) # calibrate distance
# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_1distance.csv", data, delimiter=',')

''' calibrate beam angle '''
# data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_1distance.csv"), delimiter=",")
# beta = np.poly1d([1., 1., 1., 1.]) # choose order of model(c(a) = bkl0a^2 + b1a +b2)
# beta, data = grad_desc(data, 2, beta) # cal model for distance
# data = calibrate(data, 2, beta) # calibrate distance
# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_2beam_angle.csv", data, delimiter=',')

''' calibrate incidence angle '''
# data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_2beam_angle.csv"), delimiter=",")
# beta = np.poly1d([1., 1., 1., 1., 1.]) # choose order of model(c(a) = b0a^2 + b1a +b2)
# beta, data = grad_desc(data, 3, beta) # cal model for distance
# data = calibrate(data, 3, beta) # calibrate distance
# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_3incidence_angle.csv", data, delimiter=',')
print("???")