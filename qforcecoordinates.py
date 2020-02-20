# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:04:56 2020

@author: changkyupark
"""
import numpy as np
import tools

def _theta(i, N):
    # i: i-th node
    # N: total no. of nodes
    return (i-1)*np.pi/N

def _coord(length, theta, theta_next):
    # length: length in specified axis (so la or Ca)
    # theta: i-th theta
    # theta_next: (i+1)-th theta
    return 0.5*(length/2*(1-np.cos(theta))+length/2*(1-np.cos(theta_next)))

# Coordinates of the aerodynamic load
la = 1.611
Ca = 0.505
zcoord = [] #chordwise - 81 elements
for i in np.arange(1,81):
    zcoord.append(_coord(Ca, _theta(i, 41), _theta(i+1,41)) )
    
xcoord = [] #spanwise - 41 elements
for j in np.arange(1,41):
    xcoord.append(_coord(la, _theta(j, 41), _theta(j+1,41)) )
        

# Saves aerodynamic load in an array
qdatafile = open("aerodynamicloadf100.dat", "r")
lines = qdatafile.readlines()

qforces = []
for line in lines:
    currentline = line.split(",")
    spanwise = []
    for i in currentline:
        spanwise.append(i)
    qforces.append(spanwise)
    
qforces = np.array(qforces) #chord*span (z*x) = (81*41)
    
