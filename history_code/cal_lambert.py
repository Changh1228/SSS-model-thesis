#!/usr/bin/env python

import numpy as np
import pandas as pd

# load data
data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_0.csv"), delimiter=",")

''' Calibrate delta_z '''
# Init
beta = np.ones(3) # c(a) = b0 + b1a +b2a^2
alpha = 0.2 # learing rate
tol_l = 0.01
# Normalize data
max_x = data[:,0].max() 
data[:,0] /= max_x
# prepair averaged data for loss function
data = np.hstack((data, np.zeros((len(data),1))))
for index, row in enumerate(data):
    if index %1000 ==0:
        print("row %d/%d done" % (index, len(data)))
    if row[-1] != 0:
        continue
    key = np.where((data[:,0:4]==row[0:4]).all(1))[0]
    value = data[key,4]
    mean = np.mean(value)
    data[key, 5] = mean


print('step1 done', max_x)
#data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_1.csv"), delimiter=",")
#np.savetxt('/home/chs/Desktop/Sonar/Data/drape_result/Result_1.csv', data, delimiter=',')

a = data[:, 0]
y = data[:, 4]
y_ave = data[:, 5]
# compute grad
def cal_grad(a, y, y_ave, beta):
    grad = np.zeros(3)
    catch = (beta[0] + beta[1]*a + beta[2]*a*a)*y - y_ave
    grad[0] = np.mean(y*catch)
    grad[1] = np.mean(a*y*catch)
    grad[2] = np.mean(a*a*y*catch)
    return grad

def update_beta(beta, alpha, grad):
    new_beta = beta - alpha*grad
    return new_beta

def rmse(a, y, y_ave, beta):
    catch = (beta[0] + beta[1]*a + beta[2]*a*a)*y - y_ave
    squared_err = catch*catch
    res = np.sqrt(np.mean(squared_err))
    return res

# first round
grad = cal_grad(a, y, y_ave, beta)
loss = rmse(a, y, y_ave, beta)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(a, y, y_ave, beta)

i = 1
while i<1000:
    grad = cal_grad(a, y, y_ave, beta)
    loss = rmse(a, y, y_ave, beta)
    beta = update_beta(beta, alpha, grad)
    loss_new = rmse(a, y, y_ave, beta)
    print('Round %s Diff RMSE %s, ABS RMSE %s'%(i, abs(loss_new - loss), loss_new), beta)
    i += 1
scale = np.array([max_x**2, max_x, 1])
beta_real = beta/scale
print('Done with rmse %s and beta'% abs(loss_new), beta_real)

# compute calibration result
index = 0
while index < len(data):
    if index %1000 ==0:
        print("row %d/%d done" % (index, len(data)))
    row = data[index]
    key = np.where((data[:,1:4]==row[1:4]).all(1))[0]
    value = data[key,4]
    a = data[key,0]
    Ca = beta[0] + beta[1]*a + beta[2]*a*a
    y = Ca*value
    data[index,4] = np.mean(y)
    data = np.delete(data, key[1:],axis=0)
    index += 1

a = data[:, 0]
y = data[:, 4]
y_ave = data[:, 5]
catch = y - y_ave
squared_err = catch*catch
res = np.sqrt(np.mean(squared_err))
print(res)

np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_1.csv", data, delimiter=',')
print("???")