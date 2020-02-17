# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:04:56 2020

@author: changkyupark
"""
import numpy as np

def _theta(i, N):
    # i: i-th node
    # N: total no. of nodes
    return (i-1)*np.pi/N

def _coord(length, theta, theta_next):
    # length: length in specified axis (so la or Ca)
    # theta: i-th theta
    # theta_next: (i+1)-th theta
    return 0.5*(length/2*(1-np.cos(theta))+length/2*(1-np.cos(theta_next)))

qdatafile = open("aerodynamicloadf100.dat", "r")
lines = qdatafile.readlines()

count = 0
coord = []
for line in lines:
    count += 1
    currentline = line.split(",")
    spancoord = []
    for i in currentline:
        spancoord.append(i)
    coord.append(spancoord)
coord = np.array(coord)
    