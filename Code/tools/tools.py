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

#=======================================================================================
"Integration functions for z and x direction"

def integrate_z(grid):
  """used to integrate the .dat aero data over the x-axis"""
  Ca = 0.505
  h_res = 41
  v_res = 81
  
  solution = []
  for column in range(len(grid[0])):
    A = 0
    for row in range(1,len(grid)-1):
      A += grid[row][column]/v_res*Ca
      
      A += grid[0][column]/v_res*Ca*0.5
      A += grid[v_res-1][column]/v_res*Ca*0.5
    solution.append(A)
  return solution

def integrate_x(x_list):
  """used to integrate the .dat aero data over the x-axis"""
  Ca = 0.505
  h_res = 41
  v_res = 81
  
  solution = []
  prev = 0
  value = 0
  for element in range(len(x_list)-1,-1,-1):
    value += (prev+x_list[element])/2
    prev=x_list[element]
    
    solution.append(value)
  return solution

""" How to use: """
grid = aero_data()
int_1 = integrate_z(grid)
int_2 = integrate_x(int_1)
int_3 = integrate_x(int_2)
int_4 = integrate_x(int_3)
int_5 = integrate_x(int_4)

x=np.linspace(0,1.611,41)
plt.axis([1.611,0,0,2*10**9])
plt.xlabel('x-axis')
plt.ylabel('z-axis')
plt.plot(x,int_5)
plt.show()
