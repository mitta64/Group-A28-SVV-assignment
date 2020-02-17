# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:16:23 2020

@author: Group A28 
"""
#=======================================================================================
"Cross sectional properties for bending"
"Requirement: Make it suitable for a box and an aileron cross section"
#Compute Centroid



#Compute Second Moment of Inertia

    #I_xx


    #I_yy

    
    #I_xy





#=======================================================================================
"Cross sectional properties for shear"
"Requirement: Make it suitable for a box and an aileron cross section"

#Compute Shear Centre
    # Requirements:
        # Locations of the booms
        # Skin thickness
        # Skin Locations



#Compute torsional stiffness
    
"Encode representation of cross section plus plot"

#============================================================
def integrate_z(grid, Ca = 0.505, h_res = 41, v_res = 81):
  """used to integrate the .dat aero data over the x-axis"""
  
  solution = []
  for column in range(len(grid[0])):
    A = 0
    for row in range(1,len(grid)-1):
      A += grid[row][column]/v_res*Ca
      
      A += grid[0][column]/v_res*Ca*0.5
      A += grid[v_res-1][column]/v_res*Ca*0.5
    solution.append(A)
  return solution
