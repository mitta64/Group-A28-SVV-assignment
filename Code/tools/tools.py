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
        self.x_a    = x_a       #"Distance between actuator 1 and 2 [cm]"
        self.h      = h         #"Aileron height[cm]"
        self.t_sk   = t_sk      #"Skin thickness [mm]"
        self.t_sp   = t_sp      #"Spar thickness [mm]"
        self.t_st   = t_st      #"Thickness of stiffener[mm]"
        self.h_st   = h_st      #"Height of stiffener[cm]"
        self.w_st   = w_st      #"Width of stiffener[cm]"
        self.n_st   = n_st      #"Number of stiffeners [-]"
        self.d_1    = d_1       #"Vertical displacement hinge 1[cm]"
        self.d_3    = d_3       #"Vertical displacement hinge 3[cm]"
        self.theta  = theta     #"Maximum upward deflection[deg]"
        self.P      = P         #"Load in actuator 2[kN]"

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
        self.boom_area = ((self.w_st/100) * (self.t_st/1000) 
                            + ((self.h_st - self.t_st/1000) * self.t_st))
        aileron_circumference = (((2 * math.pi * (self.h / 2)) /2) 
                                + 2 * math.sqrt((self.h /2)**2 
                                + (self.C_a - (self.h / 2))**2))
        self.boom_spacing = aileron_circumference / self.n_st
        
        
    
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


"Interpolator"
def spline_coefficient(node,value):
    # IMPORTANT: needs a grid in chronological order (from small to big)
    #This function creates a matrix containing all the splines coefficients for every node,
    #This way the main calculation only has to be done once, and spline_interpolator actually computes the value
    #input: nodes (1d list), value at these nodes (1dlist)
    #output Array containing Splinematrix

    Splinematrix = []
    for i in range(len(node)-1):
        "si = a +b(x-c)"
        a = value[i]
        b=(value[i+1]-value[i])/(node[i+1]-node[i])
        c = node[i]
        print(a,b,c)
        Splinematrix.append([a,b,c])
    return np.array(Splinematrix)

def spline_interpolator(Splinematrixx, node, inter_node):
    # This function actually interpolates (1 point)
    # input Splinematrix from previous function, all nodes (1d array), intervalue (the point to be interpolated)
    # output value function at node

    nodenumber=0
    for i in node:
        if inter_node<= node[nodenumber] or inter_node>node[-2]: #check at which spline to interpolate
            break
        else:
            nodenumber+=1
    nodenumber = nodenumber-1 #no
    a= Splinematrixx[nodenumber,0]
    b=Splinematrixx[nodenumber,1]
    c=Splinematrixx[nodenumber,2]
    si = a + b*(inter_node-c)
    return si

def plot(data, thing_to_plot, unit):
  """ Plot deflection or twist data on a 2D graph
        thing_to_plot and unit should be written as strings, like 'deflection', 'm' '"""
  x=np.linspace(0,1.611,len(data))
  plt.plot(x,data)
  plt.xlabel('span (m)')
  plt.ylabel(thing_to_plot + ' (' + unit + ')')
  plt.show()
  return()

""" how to use """ 
func = np.sin(np.linspace(0,10,100))
thing = 'deflection'
unit = 'm'
plot(func, thing, unit)

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






