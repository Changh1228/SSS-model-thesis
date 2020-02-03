#!/usr/bin/env python

import numpy as np

value = [1, 10]
mean = np.mean(value)
std = np.std(value, ddof = 1)
elem_list = filter(lambda x: abs((x-mean)/std) < 1., value)
print("???")