# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:04:56 2020

@author: changkyupark
"""
import numpy as np
import tools as tl

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
stepsize = 0.01
zcoord = [] #chordwise - 81 elements
for i in np.arange(1,81):
    zcoord.append(_coord(Ca, _theta(i, 81), _theta(i+1,81)) )
    
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



#interpolation
gridx = np.arange(0,la+stepsize,stepsize)
gridz = np.concatenate((np.arange(0, 0.505,stepsize/10),np.arange(0.505, 0,-stepsize/100)))
gridzlower = np.arange(0, 0.505,stepsize/10)
gridzupper = np.arange(stepsize, 0.505,stepsize/10)
Qforce_interpolated = []

#print(zcoord)
Qftranspose = np.transpose(qforces)
print(len(Qftranspose[0]))
ZQinterpol  = []
for zvalue in Qftranspose:
    Intermatrixzl = tl.cubic_coefficients(zcoord[:41], zvalue[:41])
    Intermatrixzu = tl.cubic_coefficients(list(reversed(zcoord[41:])), list(reversed(zvalue[41:])))
    Qsinglezlower = []
    Qsinglezupper = []
    for zcor in gridzlower:
        intervalue = tl.cubic_interpolator(Intermatrixzl, zcoord[:41], zvalue[:41], zcor)
        Qsinglezlower.append(intervalue)
    for zcor in gridzupper:
        intervalue = tl.cubic_interpolator(Intermatrixzu, list(reversed(zcoord[41:])), list(reversed(zcoord[41:])), zcor)
        Qsinglezupper.append(intervalue)
    #Qsinglezupper = list(reversed(Qsinglezupper))
    ZQinterpol.append(np.hstack((np.array(Qsinglezlower),np.array(Qsinglezupper))))



ZQinterpol = np.transpose(ZQinterpol)
print(len(ZQinterpol))
for valuex in ZQinterpol:
    Intermatrix = tl.cubic_coefficients(xcoord,valuex)
    Qsingle = []
    for xcor in gridx:
        intervalue = tl.cubic_interpolator(Intermatrix,xcoord, valuex, xcor)
        Qsingle.append(intervalue)
    Qforce_interpolated.append(Qsingle)

    
