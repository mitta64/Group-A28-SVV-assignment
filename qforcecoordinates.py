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
    # length: length in specified axis
    # theta: i-th theta
    # theta_next: (i+1)-th theta
    return 0.5*(length/2*(1-np.cos(theta))+la/2*(1-np.cos(theta_next)))
    