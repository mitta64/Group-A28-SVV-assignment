# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:16:23 2020

@author: Group A28 
"""
import math
import numpy as np
#=============================================================================
"Class containing all Aircraft data"
class Aircraft(object):
    def __init__(self,name,C_a,l_a,x_1,x_2,x_3,x_a,h,
                            t_sk,t_sp,t_st,h_st,w_st,n_st,d_1,d_3,theta,P):
        self.name   = name
        self.C_a    = C_a       #"Chord length aileron[m]"
        self.l_a    = l_a       #"Span of the aileron[m]"
        self.x_1    = x_1       #"x-location of hinge 1 [m]"
        self.x_2    = x_2       #"x-location of hinge 2 [m]"
        self.x_3    = x_3       #"x-location of hinge 3 [m]"
        self.x_a    = x_a/100   #"Distance between actuator 1 and 2 [m]"
        self.h      = h /100    #"Aileron height [m]"
        self.t_sk   = t_sk/1000 #"Skin thickness [m]"
        self.t_sp   = t_sp/1000 #"Spar thickness [m]"
        self.t_st   = t_st/1000 #"Thickness of stiffener[m]"
        self.h_st   = h_st/100  #"Height of stiffener[m]"
        self.w_st   = w_st/100  #"Width of stiffener[m]"
        self.n_st   = n_st      #"Number of stiffeners [-]"
        self.d_1    = d_1/100   #"Vertical displacement hinge 1[m]"
        self.d_3    = d_3/100   #"Vertical displacement hinge 3[m]"
        self.theta  = theta     #"Maximum upward deflection[deg]"
        self.P      = P *1000   #"Load in actuator 2 [N]"

    def description(self):
        prop = vars(self)

        for i in prop.keys():
            print(str(i)+"="+'\t'+str(prop[i]))


#=======================================================================================

    "Cross sectional properties for bending"
    "Requirement: Make it suitable for a box and an aileron cross section"
    
    """A class that computes:
        
        - Boom Areas & Locations
        - Centroid
        - Second Moment of Inertias I_xx, I_yy, I_xy
        - Shear Centre
        - Torsional Stiffness
        
        #==========================    
        Outputs: Boom Areas & Boom Locations
        Inputs: - Chord length aileron
                - Height aileron
                - Spar thickness
                - Stringer spacing delta_st
                - Number of stringers n_st
                - Stringer locations
                - Skin thickness
                - Stringer height
                - Stringer width
                - Stringer thickness 
        #==========================
        Output: Centroid
        Inputs: See input list of Boom Areas
        #==========================    
        Outputs: Second Moment of Inertias I_xx, I_yy, I_xy
        Inputs: See input list of Boom Areas
        #==========================    
        Output: Shear Centre
        Inputs: - Boom areas 
                - Boom locations 
                - Skin thickness
        #==========================    
        Output: Torsional Stiffness
        Inputs: - Shear flow distributions
        #==========================  
        Output: Visual representation of cross section
                    -> Black colour lines for the skin
                    -> Red dots for booms
        Inputs: - Aileron height
                - Aileron chord length
                - Skin thickness
                - Boom areas
                - Boom locations
        #==========================  
    """
    #========================       
    #Compute Boom Areas & Boom Locations
    #========================
    
    
    def booms(self):
        
        # Compute stringer area
        self.boom_area = ((self.w_st) * (self.t_st)                             
                            + ((self.h_st - self.t_st) * self.t_st))
        
        # Compute aileron circumference
        aileron_circumference = (((2 * np.pi * (self.h / 2)) /2)                
                                + 2 * math.sqrt((self.h /2)**2 
                                + (self.C_a - (self.h / 2))**2))
        
        # Compute boom spacing
        self.boom_spacing = aileron_circumference / self.n_st   

        # Compute orientation stringer in semi-circle & triangular section      
        angle_arc = (self.boom_spacing / (self.h / 2))                          
        angle_triangle = (math.atan((self.h /2) / (self.C_a - (self.h /2))))                                                
        
        # Start array with Col 1 z coordinate & Col 2 y coordinate 
        # Add stringers, starting at LE and going clockwise
        self.boom_loc_area = np.array([0, 0])
        
        # Add stringer in upper arc section
        boom_arc_up_loc = (np.array([                                               
                -((self.h / 2) - (np.cos(angle_arc) * (self.h / 2) )),          
                np.sin(angle_arc) * (self.h / 2)]))
        
        self.boom_loc_area = np.stack((self.boom_loc_area, boom_arc_up_loc))
        
        # Add stringers in upper triangular section
        for i in np.linspace(3.5, 0.5 ,4):
            boom_tri_up = np.array([-(self.C_a - i * self.boom_spacing * np.cos(angle_triangle)),
                                    i * self.boom_spacing * np.sin(angle_triangle)])
            self.boom_loc_area = np.append(self.boom_loc_area, [boom_tri_up], axis = 0)
            
        # Add stringers in lower triangular section
        for i in np.linspace(0.5, 3.5, 4):
            boom_tri_down = np.array([-(self.C_a - i * self.boom_spacing * np.cos(angle_triangle)),
                                    -i * self.boom_spacing * np.sin(angle_triangle)])
            self.boom_loc_area = np.append(self.boom_loc_area, [boom_tri_down], axis = 0)
        
        # Add stringer in lower arc section
        boom_arc_down_loc = (np.array([                                               
                -((self.h / 2) - (np.cos(angle_arc) * (self.h / 2) )),          
                - np.sin(angle_arc) * (self.h / 2)]))
        self.boom_loc_area = np.append(self.boom_loc_area, [boom_arc_down_loc], axis = 0)
        
        # Add column of boom areas to the total array
        boom_area_column = np.full((11,1), self.boom_area)
        self.boom_loc_area = np.append(self.boom_loc_area, boom_area_column, axis = 1)
        "Final output of booms function is self.boom_loc_area"
    # #========================       
    # #Compute Centroid
    # #========================
    # def centroid(self):
        
        
    # #========================       
    # #Compute Second Moment of Inertia
    # #========================
    # def second_moi(self):
    
    
    
    
    # #I_xx
    
    
    # #I_yy
    
    
    # #I_xy
    
    # #========================       
    # #Compute Shear Centre
    # #========================
    # # Requirements:
    #     # Locations of the booms
    #     # Skin thickness
    #     # Skin Locations
    # def shear_centre(self):
        
        
    
    # #========================       
    # #Compute Torsional Stiffness
    # #========================
    # def torsional_stiffness(self):
        
    
    
    
    
    

    

f100 = Aircraft("Fokker 100", 0.505, 1.611, 0.125, 0.498, 1.494, 24.5, 16.1, 1.1, 2.4, 1.2, 1.3, 1.7, 11, 0.389, 1.245, 30, 49.2)

#=======================================================================================
"Integration functions for z and x direction"

def macaulay(x, x_n, pwr=1):
  "returns result of the step function for [x-x_n]^pwr"
  result = (x-x_n)
  if result>=0:
    return result**pwr
  else:
    return 0


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

# """ How to use: """
# grid = aero_data()
# int_1 = integrate_z(grid)
# int_2 = integrate_x(int_1)
# int_3 = integrate_x(int_2)
# int_4 = integrate_x(int_3)
# int_5 = integrate_x(int_4)

# x=np.linspace(0,1.611,41)
# plt.axis([1.611,0,0,2*10**9])
# plt.xlabel('x-axis')
# plt.ylabel('z-axis')
# plt.plot(x,int_5)
# plt.show()


#=======================================================================================
"plotting functions"






