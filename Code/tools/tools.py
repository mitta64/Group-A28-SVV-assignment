# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:16:23 2020

@author: Group A28 
"""
import math
import numpy as np
import matplotlib.pyplot as plt
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
    def centroid(self):
        arr_z_y_a = np.zeros(shape = (3, 4 + self.n_st))

        x_circ = - 4* (self.h/2)/(3 * np.pi)
        a_circ = np.pi * self.h/2 * self.t_sk
        arr_z_y_a[:,0] = [x_circ,0.,a_circ]

        x_spr = - self.h/2
        a_spr = self.h * self.t_sp
        arr_z_y_a[:, 1] = [x_spr, 0., a_spr]

        x_sk = - (self.h/4 + self.C_a/2)
        a_sk = np.sqrt((self.h/2)**2 + (self.C_a - self.h/2)**2) * self.t_sk
        arr_z_y_a[:,2:4] = [[x_spr,x_spr], [0.,0.], [a_spr,a_spr]]

        
        
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


"Interpolators 2 different ways: linear of cubic for cubic interpolation 2 boundary conditions are required"
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
        if inter_value<= i:  #inter_value>node[-2]: #check at which spline to interpolate
            break
        if inter_value >= node[-2]:
            nodenumber = len(node)-1
            break
        else:
            nodenumber+=1
    nodenumber = nodenumber-1 #no
    a= Splinematrixx[nodenumber,0]
    b=Splinematrixx[nodenumber,1]
    c=Splinematrixx[nodenumber,2]
    si = a + b*(inter_node-c)
    return si

def cubicM_matrix(node,value,boundary1,boundary2):
    # IMPORTANT: needs a grid in chronological order (from small to big)
    #This function creates a matrix containing all the splines coefficients for every node,
    #This way the main calculation only has to be done once, and spline_interpolator actually computes the value
    #input: nodes (1d list), value at these nodes (1dlist), boundary 1 (f'(0)=?),boundary 2 (f'(n)=?)
    #output Array containing Splinematrix

    Mmatrix = []
    dmatrix = []
    Lambda0 = 1
    Mmatrix.append(list(np.concatenate((np.array([2,Lambda0]),np.zeros(len(node)-2)),axis=0)))
    dmatrix.append((((value[1]-value[0])-boundary1)/(node[1]-node[0]))/(node[1]-node[0]))
    for i in range(1,len(node)-1):
        #Main matrix
        hi      =  node[i]-node[i-1]
        hi1     =  node[i+1]-node[i]
        mui     = hi/(hi+hi1)
        Lambdai =  1-mui
        a       = np.zeros(i-1)
        if len(node)-len(a)-3>=0:
            b   = np.zeros(len(node)-len(a)-3)
        else:
            b   = np.array([])
        Mmatrix.append(list(np.concatenate((a,np.array([mui,2,Lambdai]),b),axis=0)))
        #outcome matrix
        f = ((value[i+1]-value[i])/(node[i+1]-node[i])-(value[i]-value[i-1])/(node[i]-node[i-1]))/(node[i+1]-node[i-1])
        dmatrix.append(f)
    mun=1
    Mmatrix.append(list(np.concatenate((np.zeros(len(node) - 2),np.array([mun,2])), axis=0)))
    dmatrix.append((boundary2-(value[-1] - value[-2]) / (node[-1] - node[-2])) / (node[-1] - node[-2]))
    #solve for coefficients
    dmatrix = 6*np.array(dmatrix)
    coefficients = np.linalg.solve(Mmatrix,dmatrix)
    print(coefficients)
    return coefficients

def cubic_interpolator(Mmatrixx, node, value, inter_value):
    # This function actually interpolates (1 point)
    # input Mmatrix from previous function, all nodes (1d array), intervalue (the point to be interpolated)
    nodenumber=0
    for i in node:
        if inter_value<= i:  #inter_value>node[-2]: #check at which spline to interpolate
            break
        if inter_value >= node[-2]:
            nodenumber = len(node)-1
            break
        else:
            nodenumber+=1
    nodenumber = nodenumber #no
    xi = node[nodenumber]
    x_i = node[nodenumber-1]
    yi = value[nodenumber]
    y_i = value[nodenumber-1]
    hi = xi - x_i

    a = Mmatrixx[nodenumber-1]/(6*hi)
    b =  Mmatrixx[nodenumber]/(6*hi)
    c = Mmatrixx[nodenumber-1]*hi*hi/6
    d = Mmatrixx[nodenumber]*hi*hi/6
    si = a*(xi-inter_value)**3+b*(inter_value-x_i)**3+(y_i-c)*(xi-inter_value)/hi+(yi-d)*(inter_value-x_i)/hi
    return si

#=====================================================================
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





