# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:16:23 2020
def integral(f,x1,x2,res=10000):
    i=(x2-x1)/res   # interval
    A=0
    a=f(x1)
    for e in range(res):
        b=f(x1+(e+1)*i)
        A+=(a+b)*i/2
        a=b
    return A

@author: Group A28 
"""


"Cross sectional properties for bending"
"Requirement: Make it suitable for a box and an aileron cross section"
#Compute Centroid



#Compute Second Moment of Inertia

    #I_xx


    #I_yy

    
    #I_xy






"Cross sectional properties for shear"
"Requirement: Make it suitable for a box and an aileron cross section"

#Compute Shear Centre
    # Requirements:
        # Locations of the booms
        # Skin thickness
        # Skin Locations



#Compute torsional stiffness
    
"Encode representation of cross section plus plot"

