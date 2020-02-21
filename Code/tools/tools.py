# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:16:23 2020

@author: Group A28 
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from data import aero_data, grid, transpose

#=======================================================================================
"Class containing all Aircraft data"
class Aircraft(object):
    def __init__(self,name,C_a,l_a,x_1,x_2,x_3,x_a,h,
                            t_sk,t_sp,t_st,h_st,w_st,n_st,d_1,d_3,theta,P):
        self.name   = name
        self.C_a    = C_a                   #"Chord length aileron[m]"
        self.l_a    = l_a                   #"Span of the aileron[m]"
        self.x_1    = x_1                   #"x-location of hinge 1 [m]"
        self.x_2    = x_2                   #"x-location of hinge 2 [m]"
        self.x_3    = x_3                   #"x-location of hinge 3 [m]"
        self.x_a    = round(x_a/100,8)      #"Distance between actuator 1 and 2 [m]"
        self.h      = round(h/100,8)        #"Aileron height[m]"
        self.t_sk   = round(t_sk/1000,8)    #"Skin thickness [m]"
        self.t_sp   = round(t_sp/1000,8)    #"Spar thickness [m]"
        self.t_st   = round(t_st/1000,8)    #"Thickness of stiffener[m]"
        self.h_st   = round(h_st/100,8)     #"Height of stiffener[m]"
        self.w_st   = round(w_st/100,8)     #"Width of stiffener[m]"
        self.n_st   = n_st                  #"Number of stiffeners [-]"
        self.d_1    = round(d_1/100,8)      #"Vertical displacement hinge 1[m]"
        self.d_3    = round(d_3/100,8)      #"Vertical displacement hinge 3[m]"
        self.theta  = theta                 #"Maximum upward deflection[deg]"
        self.P      = round(P*1000,8)       #"Load in actuator 2[N]"
        self.G      = 28 * 10**9            #"Shear Modulus of Aluminium 2024-T3 [Pa]"
    def description(self):

        prop = vars(self)

        for i in prop.keys():
            print(str(i) + "=" + '\t' + str(prop[i]))


#=======================================================================================

    "Cross sectional properties for bending"
    "Requirement: Make it suitable for a box and an aileron cross section"
    
    """A class that computes:
        
        - Boom Areas & Locations
        - Centroid
        - Second Moment of Inertias I_xx, I_yy, I_xy
        - Shear Centre
        - Shear flow
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
        Output: Shear flow at any given location
        Inputs: - The angle in the semi-circle
                - The y-coordinate in the spar
                - Fraction of the length of the the triangular straight line
                    from top spar to TE or from TE to bottom spar
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
            #"Final output of booms function is self.boom_loc_area"
    #=====================
    #Compute Centroid
    #=====================
    def centroid(self):
        arr_z_y_a = np.zeros(shape = (3, 4 + self.n_st))


        x_circ = - (self.h/2- self.h/np.pi)
        a_circ = np.pi * self.h/2 * self.t_sk
        arr_z_y_a[:,0] = [x_circ,0.,a_circ]

        x_spr = - self.h/2
        a_spr = self.h * self.t_sp
        arr_z_y_a[:, 1] = [x_spr, 0., a_spr]

        x_sk = - (self.h/4 + self.C_a/2)
        y_sk = self.h/4
        a_sk = np.sqrt((self.h/2)**2 + (self.C_a - self.h/2)**2) * self.t_sk
        arr_z_y_a[:,2:4] = [[x_sk,x_sk], [y_sk,-y_sk], [a_sk,a_sk]]

        arr_z_y_a[:,4:] = np.transpose(self.boom_loc_area)
        self.boom_skin_z_y_a = arr_z_y_a

        self.cent = np.round(np.array([[np.sum(arr_z_y_a[0,:]*arr_z_y_a[2,:])/np.sum([arr_z_y_a[2,:]])],[np.sum(arr_z_y_a[1,:]*arr_z_y_a[2,:])/np.sum([arr_z_y_a[2,:]])]]),5)
        
        
    #========================
    #Compute Second Moment of Inertia
    #========================
    def second_moi(self):
    #I_zz
        steiner_boom_skin_zz = np.round(np.square(self.boom_skin_z_y_a[1,:]) * self.boom_skin_z_y_a[2,:], 7)
        Izz_circ = np.pi * ((self.h/2)**4 - (self.h/2- self.t_sk)**4)/ 8
        Izz_spar = self.h**3 * self.t_sp/12
        l_sk     = np.sqrt((self.C_a-self.h/2)**2 + (self.h/2)**2)
        Izz_sk   = (l_sk)**3 * self.t_sk * ((self.h/2)/(l_sk))**2 /12

        self.Izz = np.sum(steiner_boom_skin_zz) + Izz_circ + Izz_spar + Izz_sk

    #I_yy
        steiner_boom_skin_yy = np.round(np.square(self.boom_skin_z_y_a[0,:]) * self.boom_skin_z_y_a[2,:], 7)

        Iyy_circ = (np.pi/8 - 8/(np.pi * 9)) * ((self.h / 2) ** 4 - (self.h / 2 - self.t_sk) ** 4)
        Iyy_spar = 0
        l_sk     = np.sqrt((self.C_a - self.h / 2) ** 2 + (self.h / 2) ** 2)
        Iyy_sk   = (l_sk) ** 3 * self.t_sk * ((self.C_a - self.h / 2) / (l_sk))** 2 / 12

        self.Iyy = np.sum(steiner_boom_skin_yy) + Iyy_circ + Iyy_spar + Iyy_sk


    #I_xy
        self.Iyz = 0.
    
    #========================       
    #Compute Shear Centre
    #========================
    # Requirements:
        # Locations of the booms
        # Skin thickness
        # Skin Locations
    def shear_centre(self):
        
        # Radius of semi-cirle
        h = self.h / 2
        # Length of triangular section
        L_sk = math.sqrt((self.C_a - h)**2 + h**2)
        # Angle at TE of ONE triangular section
        alpha = math.atan((h) / (self.C_a - (h)))
        # Self.boom_loc_area becomes "a" for simplicity
        a = self.boom_loc_area
        
        # Shear flows [N/m]
        # Shear flow in bottom triangular section
        qb_3 = (-1 / self.Izz) * (-self.t_sk * h * (L_sk/2) +
                self.boom_area * a[6,1] + self.boom_area * a[7,1] +
                self.boom_area * a[8,1] + self.boom_area * a[9,1])
        # Shear flow in bottom part spar
        qb_4 = (-1 / self.Izz) * (self.t_sp * h**2 / 2 )
        # Shear flow in semi-circle
        qb_5 = (-1 / self.Izz) * (-self.t_sk * h * (L_sk/2) +
                - self.t_sp * h**2 / 2 + self.boom_area * a[1,1] +
                self.boom_area * a[6,1] + self.boom_area * a[7,1] +
                self.boom_area * a[8,1] + self.boom_area * a[9,1] +
                self.boom_area * a[10,1])
        # Shear flow in top part spar
        qb_1 = qb_4
        # Shear flow in upper triangular section
        qb_2 = (-1 / self.Izz) * (self.boom_area * a[1,1] + 
                self.boom_area * a[2,1] + self.boom_area * a[3,1] +
                self.boom_area * a[4,1] + self.boom_area * a[5,1] +
                self.boom_area * a[6,1] + self.boom_area * a[7,1] +
                self.boom_area * a[8,1] + self.boom_area * a[9,1] +
                self.boom_area * a[10,1])
        # Redundant shear flow in left cell
        q0_1 = ((-1) * ((qb_5 * np.pi * h)/(self.G * self.t_sk) - 
                (qb_1 * h)/(self.G * self.t_sp) -
                (qb_4 * h)/(self.G * self.t_sp))) /((np.pi * h)/(self.G * self.t_sk)
                                                    + (2 * h)/(self.G * self.t_sp))
        q0_2 = ((-1) * ((qb_1 * h)/(self.G * self.t_sp) +
                (qb_2 * L_sk)/(self.G * self.t_sk) + 
                (qb_3 * L_sk)/(self.G * self.t_sk) +
                (qb_4 * h)/(self.G * self.t_sp))) /((2 * h)/(self.G * self.t_sp)
                                                    + (2 * L_sk)/(self.G * self.t_sk))
        
        # Shear Centre z and y location (due to symmetry y = 0)
        self.shear_centre_z = (-1) * ((qb_5 * np.pi * h**2) +
                                      (qb_2 * L_sk * np.cos(alpha) * h) +
                                      (qb_3 * L_sk * np.cos(alpha) * h) +
                                      (q0_1 * np.pi * h**2) +
                                      (q0_2 * 2 * h * (self.C_a - h))) - h
                                
        self.shear_centre_y = 0                                          
    #================================ 
    #Compute Shear Flow At Any Point
    #=================================
    # Input: Angle theta [rad] from -pi/2 to pi/2
    # Input: y-coordinate [m] from -self.h/2 to self.h/2
    # Input: Fraction of length skin L_sk    
    def master_shear_flow(self, theta=0, y=-0.05, frac=0):
        
        # Radius of semi-cirle
        h = self.h / 2
        # Length of triangular section
        L_sk = math.sqrt((self.C_a - h)**2 + h**2)
        # Self.boom_loc_area becomes "a" for simplicity
        a = self.boom_loc_area
        
        # Input: Angle theta [rad] from -pi/2 to pi/2
        # Semi-cirlce shear flow
        self.qb_5 = (-1 / self.Izz) * (-self.t_sk * h**2 * np.cos(theta) -self.t_sk * h * (L_sk/2) +
                - self.t_sp * h**2 / 2 + self.boom_area * a[1,1] +
                self.boom_area * a[6,1] + self.boom_area * a[7,1] +
                self.boom_area * a[8,1] + self.boom_area * a[9,1] +
                self.boom_area * a[10,1])
        
        # Lower part spar shear flow
        # Input: y-coordinate [m] from -self.h/2 to 0
        self.qb_4 = (-1 / self.Izz) * (self.t_sp * y**2 / 2 ) 
        
        # Upper part spar shear flow
        # Input: y-coordinate [m] from 0 to self.h/2
        self.qb_1 = self.qb_4
        
        # Lower part triangle shear flow
        # Input: Fraction of length skin L_sk
        self.qb_3 = (-1 / self.Izz) * (-self.t_sk * h * (1/(2 * L_sk)) * (frac * L_sk)**2 +
                self.boom_area * a[6,1] + self.boom_area * a[7,1] +
                self.boom_area * a[8,1] + self.boom_area * a[9,1])
        
        # Upper part triangle shear flow
        # Input: Fraction of length skin L_sk
        self.qb_2 = (-1 / self.Izz) * (self.t_sk * h * ((frac*L_sk) - ((frac * L_sk)**2) / (2 * L_sk))
                - self.t_sk * h * (L_sk / 2) + self.boom_area * a[1,1] + 
                self.boom_area * a[2,1] + self.boom_area * a[3,1] +
                self.boom_area * a[4,1] + self.boom_area * a[5,1] +
                self.boom_area * a[6,1] + self.boom_area * a[7,1] +
                self.boom_area * a[8,1] + self.boom_area * a[9,1] +
                self.boom_area * a[10,1])
    
    #========================       
    #Compute Torsional Stiffness
    #========================
    # Apply unit torque -> T = 1
    # Set up torque equation and dtheta/dz equations for cell I and II
    # Solve for q0_2, q0_1 and dtheta_dz
    # Obtain torsional stiffness J from T/(G * dtheta/dz)
    def torsional_stiffness(self):
        
        # Radius of semi-cirle
        h = self.h / 2
        # Length of triangular section
        L_sk = math.sqrt((self.C_a - h)**2 + h**2)
        
        # Compute q0_2, q0_1, dtheta_dz and self.J
        # A = X * q0_2
        A = (2 * (self.C_a - h)) / (h * self.G * self.t_sk)
        + (4 * (self.C_a - h)) / (np.pi * h * self.G * self.t_sp)
        + (2) / (self.G * self.t_sp)
        
        X = (2 * np.pi * h * L_sk) / (self.G * self.t_sk)
        + (2 * np.pi * h**2) / (self.G * self.t_sp) 
        + (4 * h * (self.C_a - h)) / (self.G * self.t_sp)
        + ((2 * h * (self.C_a - h))**2) / ((h)**2 * self.G * self.t_sk)
        + (8 * (h * (self.C_a - h))**2) / (np.pi * (h)**2 * self.G * self.t_sp)
        + (4 * h * (self.C_a - h)) / (self.G * self.t_sp)
        
        q0_2 = A / X 
        
        # From Torque equation obtained        
        q0_1 = (1 - (2 * h * (self.C_a - h)) * q0_2) / (np.pi * (h)**2) 
        
        # Cell I dtheta_dz used
        dtheta_dz = (1 / (np.pi * h)) * ((q0_1 * np.pi) / (self.G * self.t_sk)
                                         + (2 * (q0_1 - q0_2)) / (self.G * self.t_sp))
        # Torsional stiffness J
        self.J = 1 / (self.G * dtheta_dz)
        
        
#=======================================================================================
f100 = Aircraft("Fokker 100", 0.505, 1.611, 0.125, 0.498, 1.494, 24.5, 16.1, 1.1, 2.4, 1.2, 1.3, 1.7, 11, 0.389,
                    1.245, 30, 49.2)


def macaulay(x, x_n, pwr=1):
  "returns result of the step function for [x-x_n]^pwr"
  result = (x-x_n)
  if result>=0:
    return result**pwr
  else:
    return 0

def matrix(alpha,h, x_1, x_2, x_3, x_a,I,E):
    """Constructs the matrix A such that Ax=b for the statically indeterminate
    problem. Where:
    A is the matrix
    x = (R_1y, R_2y, R_3y, R_1z, R_2z, R_3z, R_i, C_1, C_2, C_3, C_4, C_5)
    b = ()
    Inputs:
    Section = ('z':I_zz, 'y':I_yy, 'G':G, 'J':J, 'E':E, 'z_sc':z_sc)""" 
    Ky    = (1/(I['E']*I['y']))
    Kz    = (1/(I['E']*I['z']))
    L     = 1/(I['G']*I['J'])
    Ksi_1 = x_2-x_a/2
    Ksi_2 = x_2+x_a/2
    Eta   = h/2 + z_sc
    mc = macaulay

    def Alpha(a,b):
    #helper function
        return   (-(Kz*np.sin(alpha)/6 *mc(a,b,3) +
            L*Eta*z_sc*np.sin(alpha)   *mc(a,b) + 
            L*Eta*h/2 *np.cos(alpha)   *mc(a,b)))

    def Gamma(a,b):
    #helper function
        return Kz/6 * mc(a, b, 3) - L*Eta**2*mc(a, b)

    #       x =#(               R_1y,              R_2y,              R_3y,                                 R_1z,                                 R_2z,                                 R_3z,                                  R_i,                   C_1,           C_2,                   C_3,           C_4,                                 C_5)
    A = np.array([[                1,                 1,                 1,                                    0,                                    0,                                    0,                        np.sin(alpha),                     0,             0,                     0,             0,                                   0],#Row 1
                  [                0,                 0,                 0,                                    1,                                    1,                                    1,                        np.cos(alpha),                     0,             0,                     0,             0,                                   0],#Row 2
                  [             -h/2,              -h/2,              -h/2,                                    0,                                    0,                                    0, -h/2 * (np.sin(alpha)+np.cos(alpha)),                     0,             0,                     0,             0,                                   0],#Row 3
                  [                0,                 0,                 0,                                  x_1,                                  x_2,                                  x_3,            np.cos(alpha)*(x_2-x_a/2),                     0,             0,                     0,             0,                                   0],#Row 4
                  [             -x_1,              -x_2,              -x_3,                                    0,                                    0,                                    0,           -np.sin(alpha)*(x_2-x_a/2),                     0,             0,                     0,             0,                                   0],#Row 5
                  [                0,   Gamma(x_1, x_2),   Gamma(x_1, x_3),                                    0,                                    0,                                    0,                    Alpha(x_1, Ksi_1),                   x_1,             1,                     0,             0,                                   1],#Row 6
                  [                0,                 0,                 0,                                    0,                 Ky/6*mc(x_1, x_2, 3),                 Ky/6*mc(x_1, x_2, 3), Ky*np.cos(alpha)/6 *mc(x_1, Ksi_1,3),                     0,             0,                   x_1,             1,                                   0],#Row 7
                  [  Gamma(x_2, x_1),                 0,   Gamma(x_2, x_3),                                    0,                                    0,                                    0,                    Alpha(x_2, Ksi_1),                   x_2,             1,                     0,             0,                                   1],#Row 8
                  [                0,                 0,                 0,                 Ky/6*mc(x_2, x_1, 3),                                    0,                 Ky/6*mc(x_2, x_2, 3), Ky*np.cos(alpha)/6 *mc(x_2, Ksi_1,3),                     0,             0,                   x_2,             1,                                   0],#Row 9
                  [  Gamma(x_3, x_1),   Gamma(x_3, x_2),                 0,                                    0,                                    0,                                    0,                    Alpha(x_3, Ksi_1),                   x_3,             3,                     0,             0,                                   1],#Row 10
                  [                0,                 0,                 0,                 Ky/6*mc(x_3, x_1, 3),                 Ky/6*mc(x_3, x_2, 3),                                    0, Ky*np.cos(alpha)/6 *mc(x_3, Ksi_1,3),                     0,             0,                   x_3,             1,                                   0],#Row 11
                  [Alpha(Ksi_1, x_1), Alpha(Ksi_1, x_2), Alpha(Ksi_1, x_3), Ky*np.cos(alpha)/6 *mc(Ksi_1, x_1,3), Ky*np.cos(alpha)/6 *mc(Ksi_1, x_2,3), Ky*np.cos(alpha)/6 *mc(Ksi_1, x_2,3),                                    0, Ksi_1 * np.sin(alpha), np.sin(alpha), Ksi_1 * np.cos(alpha), np.cos(alpha), z_sc*(np.sin(alpha)+np. cos(alpha))]#Row 1 2
        ])
    b = {}
    

#=======================================================================================
"Integration functions for z and x direction"

def def_integral(f,x1,x2,res=10000):
    interval = (x2-x1)/res
    solution = 0
    a=f(x1)
    for e in range(res):
        b=f(x1+(e+1)*interval)
        solution += (a+b)*interval/2
        a=b
    return solution

def indef_integral(f,x1,x2,res=10000):
    interval = (x2-x1)/res
    solution = []
    value = 0
    a = f(x1)
    for e in range(res):
        b = f(x1+(e+1)*interval)
        value += (a+b)*interval/2
        solution.append(value)
        a = b
    return solution


""" This calculates the n'th integral (with minimum of n=1). It is structured so that the program first calculates the definite integral from z=0 till z=C_a= -0.505.
Then, it calculates the indeffinite integral along dx. The n'th integral (if n>=2) will than be the definite integral for x=0 till x=l_a=1.611
res is the resolution. Higher value = more accurate, but longer runtime """
def integral_z(n,x_final=1.611,res=1000):
    #--------------------- input data --------------------------------
    """ boundaries of the integration """
    x1 ,x2 = 0, 1.611
    z1, z2 = 0, 0.505

    #------------------ main program ---------------------------
    start_time = time.time() # to calculate runtime of the program

    """ The program can only calculate integrals of functions, not matrixes or wathever.
    This function can only have one variable as input: x-value. It also outputs only one value: y-value (=interpolated aero_data)
    The following defenitinion makes such a function that can later be used in the integral"""
    def function(x):
        y = spline_interpolator(matrix, nodes, x)
        return y


    """ the function 'spline_coefficient(nodes,row)' converts an array of x-values (=nodes) and an array of y-values (=column of the aero_data) into a matrix. This matrix is necessary to use the function 'spline_interpolator'. (see interpolation file for explenation) """
    nodes = np.linspace(z1,z2,len(grid[0]))
    solution = []
    for row in grid:
        matrix = spline_coefficient(nodes, row)
        """ This calculates the definite integral from z1 to z2 of 'function' """
        a = def_integral(function,z1,z2,res)
        solution.append(a)
        """ The result is a 1D array of data corresponding to the values of the definite integrals of interpolated columns of the aero_data """

    if n > 2:
        for i in range(n-2):
            nodes = np.linspace(x1,x2,len(solution))
            matrix = spline_coefficient(nodes, solution)
            solution = indef_integral(function,x1,x2,res)
            
    """ This can be used to check the results for when n=1 (only integrated once w.r.t. z-axis) or an intermediate step of another integration"""
    plot_to_show = 2   # Show the plot of the n'th integral. plot_to_show = 0 for no plots.
    if n == 1 or n-1==plot_to_show:
        x = np.linspace(0,1.611,len(solution))
        plt.xlabel('x-axis')
        plt.ylabel('z-axis')
        plt.plot(x,solution)
        plt.show()

    if n > 1:
        nodes = np.linspace(x1,x2,len(solution))
        matrix = spline_coefficient(nodes, solution)
        solution = def_integral(function,x1,x_final,res)


    end_time = time.time()
    run_time = end_time - start_time   # print run_time to see the time it took the program to compute
    return solution




def integral_x(n,z_final=0.505,res=1000):
    #--------------------- input data --------------------------------
    x1 ,x2 = 0, 1.611
    z1, z2 = 0, 0.505

    #------------------ main program ---------------------------
    def function(x):
        y = spline_interpolator(matrix, nodes, x)
        return y


    nodes = np.linspace(x1,x2,len(aero_data[0]))
    solution = []
    for row in aero_data:
        matrix = spline_coefficient(nodes, row)
        a = def_integral(function,x1,x2,res)
        solution.append(a)

    if n > 2:
        for i in range(n-2):
            nodes = np.linspace(z1,z2,len(solution))
            matrix = spline_coefficient(nodes, solution)
            solution = indef_integral(function,z1,z2,res)
            

    if n > 1:
        nodes = np.linspace(z1,z2,len(solution))
        matrix = spline_coefficient(nodes, solution)
        solution = def_integral(function,z1,z_final,res)

    return solution

#=======================================================================================
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
        # print(a,b,c)
        Splinematrix.append([a,b,c])
    return np.array(Splinematrix)

def spline_interpolator(Splinematrixx, node, inter_node):
    # This function actually interpolates (1 point)
    # input Splinematrix from previous function, all nodes (1d array), intervalue (the point to be interpolated)
    # output value function at node

    nodenumber=0
    for i in node:
        if inter_node<= i:  #inter_value>node[-2]: #check at which spline to interpolate
            break
        if inter_node >= node[-2]:
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

def cubic_coefficients(node,value):
    # IMPORTANT: needs a grid in chronological order (from small to big)
    #This function creates a matrix containing all the splines coefficients for every node,
    #This way the main calculation only has to be done once, and spline_interpolator actually computes the value
    #input: nodes (1d list), value at these nodes (1dlist), boundary 1 (f'(0)=?),boundary 2 (f'(n)=?)
    #output Array containing Splinematrix
    boundary1 = (value[1]-value[0])/(node[1]-node[0])
    boundary2 = (value[-1]-value[-2])/(node[-1]-node[-2])
    # print(boundary1,boundary2)
    Mmatrix = []
    dmatrix = []
    Lambda0 = 1
    #boundary 1
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
    #boundary
    Mmatrix.append(list(np.concatenate((np.zeros(len(node) - 2),np.array([mun,2])), axis=0)))
    dmatrix.append((boundary2-(value[-1] - value[-2]) / (node[-1] - node[-2])) / (node[-1] - node[-2]))
    #solve for coefficients
    dmatrix = 6*np.array(dmatrix)
    coefficients = np.linalg.solve(Mmatrix,dmatrix)
    return coefficients

def cubic_interpolator(coefficients, node, value, inter_node):
    # This function actually interpolates (1 point)
    # input Splinematrix from previous function, all nodes (1d array), intervalue (the point to be interpolated)
    nodenumber=0
    for i in node:
        if inter_node<= i:  #inter_value>node[-2]: #check at which spline to interpolate
            break
        if inter_node >= node[-2]:
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

    a = coefficients[nodenumber-1]/(6*hi)
    b =  coefficients[nodenumber]/(6*hi)
    c = coefficients[nodenumber-1]*hi*hi/6
    d = coefficients[nodenumber]*hi*hi/6
    si = a*(xi-inter_node)**3+b*(inter_node-x_i)**3+(y_i-c)*(xi-inter_node)/hi+(yi-d)*(inter_node-x_i)/hi
    return si


#=====================================================================
"plotting functions"
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
#plot(func, thing, unit)

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





