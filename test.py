#!/usr/bin/env python

import numpy as np
from multiprocessing import Pool

def task(ipt):
    result = []
    idx = []
    for i in range(10000):
        result.append(ipt+10*i)
        idx.append(str(ipt+10*i))
    return (result, idx)

data_result = []
idx_result = []
def call_back(result):
    (value, idx) = result
    for i in range(len(value)):
        data_result.append(value[i])
        idx_result.append(idx[i])



data = np.array([0,1,2,0,2,1,3,4,5,3,5,6,7,0,1,2,0,2,1,3,4,5,3,5,6,7])
p = Pool(8)
for i in range(len(data)):
    p.apply_async(task, args=(data[i]),callback = call_back) #.get() 
p.close()
p.join()
for i in range(len(idx_result)):
    if int(idx_result[i]) != data_result[i]:
        print("???")
        break
