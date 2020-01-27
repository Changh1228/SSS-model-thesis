#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("/home/chs/Desktop/Sonar/Data/drape_result/Result_beam_angle.csv"), delimiter=",")

C_dist = np.poly1d([1.96541845e-07, 3.16217568e-05, 4.82854645e-03, 7.09456394e-01])
C_beam = np.poly1d([0.1180179 ,  0.13541105,  0.27383269, -0.13707748,  0.78609771])
C_incident = np.poly1d([0.11158967, 0.14816708, 0.22780308, 0.03783167, 0.67531576])

index = 0
while index < len(data):
    if index %1000 ==0:
        print("row %d/%d done" % (index, len(data)))
    row = data[index]
    result = row[4]/C_incident(row[3])
    # print(C_dist(row[1])*C_beam(row[2])*C_incident(row[3]))
    data[index,4] = result
    index += 1

beam_angle = data[:,2]
incidence = data[:,3]
indensity = data[:,4]
distance = data[:,1]
plt.figure(0)
plt.grid()
plt.title("beam_angle-intensity")
plt.scatter(beam_angle, indensity, s=2, c='b', marker='.')
plt.show()

# np.savetxt("/home/chs/Desktop/Sonar/Data/drape_result/Result_???.csv", data, delimiter=',')

