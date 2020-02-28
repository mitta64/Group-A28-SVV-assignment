# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:16:23 2020

@author: Group A28 
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from data import aero_data, grid, nodes_z, nodes_x, times_z, transpose
from integrator import def_integral, indef_integral
from interpolator import spline_coefficient, spline_interpolator
from interpolator import cubic_interpolator, cubic_coefficients
import copy

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
        self.P      = round(P*1000,8)           #"Load in actuator 2[N]"
        # Material properties
        self.G      = 28 * 10**9            #"Shear Modulus of Aluminium 2024-T3 [Pa] is 28"
        self.E      = 73.1 * 10**9          #"Elasticity Modulus of Aluminium 2024-T3 [Pa] is 71.1"
        self.booms()
        self.centroid()
        self.second_moi()
        #f100.shear_centre()
        self.torsional_stiffness()

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
        aileron_circumference = (((2 * np.pi * (self.h / 2)) / 2)
                                 + 2 * math.sqrt((self.h / 2) ** 2
                                                 + (self.C_a - (self.h / 2)) ** 2))

        # Compute boom spacing
        self.boom_spacing = aileron_circumference / self.n_st

        # Compute orientation stringer in semi-circle & triangular section
        self.angle_arc = (self.boom_spacing / (self.h / 2))
        angle_triangle = (math.atan((self.h / 2) / (self.C_a - (self.h / 2))))

        # Start array with Col 1 z coordinate & Col 2 y coordinate
        # Add stringers, starting at LE and going clockwise
        self.boom_loc_area = np.zeros(shape=(self.n_st, 3))
        # calc amount of stringers in arc
        self.n_arc_half = int((np.pi / 2) / self.angle_arc)
        # print('n_arc', self.n_arc_half)
        # Add stringer in 0,0 arc
        self.boom_loc_area[0, :] = np.array([0, 0, self.boom_area])

        # Add stringer in upper arc section
        for i in np.arange(1, self.n_arc_half + 1, 1, dtype=int):
            boom_arc_up = np.array([-((self.h / 2) - (np.cos(self.angle_arc * i) * (self.h / 2))),
                                    np.sin(self.angle_arc * i) * (self.h / 2), self.boom_area])
            self.boom_loc_area[i, :] = boom_arc_up

            # boom_arc_down = (np.array([
            # -((self.h / 2) - (np.cos(self.angle_arc) * (self.h / 2))),
            # - np.sin(self.angle_arc) * (self.h / 2), self.boom_area]))

            # self.boom_loc_area[i+1,:] = boom_arc_down
            # print('upper arc',i, boom_arc_up)

        # Add stringers in upper triangular section

        pos = self.n_arc_half + 1
        for i in np.arange((self.n_st - (self.n_arc_half * 2 + 1)) / 2 - 0.5, -0.5, -1):
            boom_tri_up = np.array([-(self.C_a - i * self.boom_spacing * np.cos(angle_triangle)),
                                    i * self.boom_spacing * np.sin(angle_triangle), self.boom_area])
            self.boom_loc_area[pos, :] = boom_tri_up

            # print('upper tri',pos,boom_tri_up,i)
            pos = pos + 1

        # Add stringers in lower triangular section
        for i in np.arange(0.5, (self.n_st - (self.n_arc_half * 2 + 1)) / 2 + 0.5, 1):
            boom_tri_down = np.array([-(self.C_a - i * self.boom_spacing * np.cos(angle_triangle)),
                                      -i * self.boom_spacing * np.sin(angle_triangle), self.boom_area])
            self.boom_loc_area[pos] = boom_tri_down

            # print('lower tri',pos,boom_tri_down,i)
            pos = pos + 1

        # Add in lower arc section

        for i in np.arange(self.n_arc_half, 0, -1, dtype=int):
            boom_arc_down = (np.array([
                -((self.h / 2) - (np.cos(self.angle_arc * (i)) * (self.h / 2))),
                - np.sin(self.angle_arc * (i)) * (self.h / 2), self.boom_area]))

            self.boom_loc_area[pos, :] = boom_arc_down
            # print('lower arc',pos,boom_arc_down)
            pos = pos + 1


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
        # I_zz
        steiner_boom_skin_zz    = np.square(self.boom_skin_z_y_a[1, :]) * self.boom_skin_z_y_a[2, :]
        Izz_circ = np.pi * ((self.h / 2) ** 4 - (self.h / 2 - self.t_sk) ** 4) / 8
        Izz_spar = self.h ** 3 * self.t_sp / 12
        l_sk     = np.sqrt((self.C_a - self.h / 2) ** 2 + (self.h / 2) ** 2)
        Izz_sk   = (l_sk) ** 3 * self.t_sk * ((self.h / 2) / (l_sk)) ** 2 / 12

        self.Izz = np.sum(steiner_boom_skin_zz) + Izz_circ + Izz_spar + Izz_sk * 2

        # I_yy
        steiner_boom_skin_yy = np.square(self.boom_skin_z_y_a[0, :] - self.cent[0]) * self.boom_skin_z_y_a[2, :]

        Iyy_circ = (np.pi / 8 - 8 / (np.pi * 9)) * ((self.h / 2) ** 4 - (self.h / 2 - self.t_sk) ** 4)
        Iyy_spar = 0
        Iyy_sk = (l_sk) ** 3 * self.t_sk * ((self.C_a - self.h / 2) / (l_sk)) ** 2 / 12

        self.Iyy = np.sum(steiner_boom_skin_yy) + Iyy_circ + Iyy_spar + Iyy_sk * 2

        # I_xy
        self.Iyz = 0.
    #========================       
    #Compute Shear Centre
    #========================

    def cell_lists(self):
        qb = self.flow_from_shear()

        spar_locations = np.where(qb == self.h/2)[1]

        cell_1 = self.qb[:,5:10] #From spar cut along circ to spar cut
        cell_2 = np.hstack((qb[:,0:spar_locations[0]+1], #from TE cut via lower to spar cut
                            qb[:,spar_locations[1]:]
                          ))#from spar cut via upper to TE cut
        
        return (cell_1, cell_2)

        
    def flow_from_shear(self,V=1):
        """Finds shear flow due to force V without q0,1/2 condistribution"""
        h       = self.h / 2                                # Radius of semi-cirle
        L_sk    = math.sqrt((self.C_a - h) ** 2 + h ** 2)   # Length of triangular section
        alpha   = math.atan((h) / (self.C_a - (h)))         # Angle at TE of ONE triangular section
        a       = self.boom_loc_area                        # Self.boom_loc_area becomes "a" for simplicity

        # Shear flows [N/m]

        # Define theta values
        theta_7         = (np.pi / 2 - self.angle_arc * self.n_arc_half)
        theta_9         = np.pi - theta_7
        booms_triangle  = int((self.n_st - (2 * self.n_arc_half + 1)) / 2)
        s12             = L_sk - (booms_triangle - 0.5) * self.boom_spacing
            # Number of booms in the bottom triangle,
            # which is equal to the number of booms in the top triangle

        

        # Shear flow bottom triangle
        start   = int(((self.n_st - 1) / 2))
        self.qb = np.zeros(shape=(2, (self.n_st + 5)))
                    #First row codes shear flow
                    #Second row codes web length
        qb_1    = (-V / self.Izz) * ((-self.t_sk * h * (0.5 * self.boom_spacing)**2) / (2 * L_sk))
        
        self.qb[0, 0]   = qb_1
        self.qb[1, 0]   = self.boom_spacing/2
        boom_con        = self.boom_area * a[start + 1, 1]
        
        for i in range(1, booms_triangle):
            self.qb[0, i] = (-V / self.Izz) * (
                    -self.t_sk * (h / L_sk) * ((self.boom_spacing * (1 + i * 2) / 2) ** 2 / 2) + boom_con)
            self.qb[1, i] = self.boom_spacing
            boom_con += self.boom_area * a[start + i + 1, 1]
            
        qb_5 = (-V / self.Izz) * (-self.t_sk * (h / L_sk) * ((L_sk ** 2) / 2) + boom_con)
        self.qb[0, booms_triangle] = qb_5
        self.qb[1, booms_triangle] = self.boom_spacing/2

        # Shear flow spar
        qb_6 = (-V / self.Izz) * (-(self.t_sp * (h) ** 2) / (2))
        self.qb[0, booms_triangle + 1]  = qb_6
        self.qb[1, booms_triangle + 1]  = h

        # Shear flow semi-circle
        qb_7 = (-V / self.Izz) * (self.t_sk * (h) ** 2 * (-np.cos(theta_7 - np.pi / 2))) + qb_5 + qb_6
        self.qb[0, booms_triangle + 2] = qb_7
        self.qb[1, booms_triangle + 2] = self.boom_spacing / 2
        
        runs = 1
        start = start + booms_triangle + 1
        for i in range(booms_triangle + 3, booms_triangle + 3 + self.n_arc_half):
            self.qb[0, i] = ((-V / self.Izz) * 
                (self.t_sk * (h) ** 2 * 
                    (-np.cos(theta_7 - np.pi / 2 + runs * self.angle_arc) + 
                        np.cos(theta_7 - np.pi / 2 + (runs - 1) * self.angle_arc)) 
                + self.boom_area * a[start, 1]
                ) + self.qb[0, i - 1])
            self.qb[1, i] = self.boom_spacing
            runs += 1

        start = 0
        runs = 1
        for i in range(booms_triangle + 3 + self.n_arc_half, booms_triangle + 3 + 2 * self.n_arc_half):
            self.qb[0, i] = (-V / self.Izz) * (self.t_sk * (h) ** 2 * (
                    -np.cos(theta_7 - np.pi / 2 + (runs + 1) * self.angle_arc) + np.cos(
                theta_7 - np.pi / 2 + (runs) * self.angle_arc)) + self.boom_area * a[start, 1]) + self.qb[0, i - 1]
            start   += 1
            runs    += 1

        qb_10 = (-V / self.Izz) * (
                self.t_sk * (h) ** 2 * (np.cos((np.pi / 2) - theta_9)) + self.boom_area * a[start, 1]) + self.qb[
                    0, booms_triangle + 3 + 2 * self.n_arc_half - 1]
        self.qb[0, booms_triangle + 3 + 2 * self.n_arc_half] = qb_10
        start += 1
        
        # Shear flow spar
        qb_11 = (-V / self.Izz) * (self.t_sp * (h ** 2) / 2)
        self.qb[0, booms_triangle + 3 + 2 * self.n_arc_half + 1] = qb_11
    
        # Shear flow top triangle
        qb_12 = (-V / self.Izz) * (self.t_sk * h * (s12 - ((s12) ** 2 / (2 * L_sk)))) + qb_11 + qb_10
        self.qb[0, booms_triangle + 3 + 2 * self.n_arc_half + 2] = qb_12

        run = 1
        boom_con = self.boom_area * a[start, 1]
        for i in range(booms_triangle + 3 + 2 * self.n_arc_half + 3,
                       2 * booms_triangle + 3 + 2 * self.n_arc_half + 2, ):
            upper = s12 + run * self.boom_spacing
            self.qb[0, i] = (-V / self.Izz) * (self.t_sk * h * (
                    (upper) - (upper) ** 2 / (2 * L_sk))
                                               + boom_con) + qb_11 + qb_10
            start += 1
            boom_con += self.boom_area * a[start, 1]
            run += 1

        qb_16 = (-V / self.Izz) * (self.t_sk * h * (L_sk / 2) + boom_con) + qb_11 + qb_10
        self.qb[0, 2 * booms_triangle + 3 + 2 * self.n_arc_half + 2] = qb_16
        
        self.qb[1]+=self.qb[1][::-1]

        #return self.qb


    #def shear_centre(self):
    
    def dtheta_dx(self):
        n_booms_triangle = int((self.n_st - (2 * self.n_arc_half + 1)) / 2)
        qb               = self.flow_from_shear()
        n_steps_circ     = (self.n_arc_half*2+1)+1  #number of steps in semicircle
        

        # Cell 1 (circle) going from the cut downwards
        Area_1 = np.pi *(self.h/2)**2 /2
        int_start_pos = n_booms_triangle + 2

        # spa_ar 1st half
        int_qb_1   = (self.qb[0,n_booms_triangle+2]*self.h/2)/(2*self.t_sp)  #line integral of qb/t
        int_qs_circ_01       = self.h/(2*self.t_sp)
        int_qs_circ_02       = - self.h/(2*self.t_sp)

        # cir
        q_start     = qb[0, int_start_pos - 1] + qb[0, int_start_pos - 2]  # shear flow at begining of integration
        int_qb_circ = (q_start + qb[0, int_start_pos]) / 2 * qb[1, int_start_pos] * self.h / (2*self.t_sk)  # integration start
        
        for i in range(int_start_pos, int_start_pos + n_steps_circ):
            int_qb_circ += (qb[0, i] + qb[0, i + 1]) / 2 * qb[1, i + 1] /(self.t_sk)

        int_qs_circ_01 += np.pi * (self.h/2)**2/(2*self.t_sk)

         # spar 2nd half
        int_qb_circ        += (self.qb[0, n_booms_triangle + 2] * self.h / 2) / (2 * self.t_sp)  # line integral of qb/t
        int_qs_circ_01      += self.h / (2 * self.t_sp) #sign flipped for triangular side
        int_qs_circ_02      += - self.h / (2 * self.t_sp)

        #Cell 2 (Triangle) Going from cut downwards:
        #Area_2 = (self.C_a-h)*h

        int_qb_tri      = 0
        q_curr          = 0
        q_last          = 0

        for i in range(n_booms_triangle):
            q_curr           = (self.qb[0,i] + q_last) /2
            int_qb_tri      += q_curr*self.qb[1,i]/ (self.t_sk)
            q_last           = self.qb[0,i]

        q_curr          = 0
        q_last          = 0

        n_start         = -n_booms_triangle-1
        int_qb_tri   += (self.qb[0,n_start-2]+
                            self.qb[0,n_start-1]+
                            self.qb[0,n_start])*self.qb[1,n_start]/(2*self.t_sk)
        n_start         += 1

        for i in range(len(self.qb[0])+n_start,len(self.qb[0])):
            print(self.qb[0,i],self.qb[0,i-1],self.qb[1,i])
            int_qb_tri   += (self.qb[0,i]+
                                self.qb[0,i-1])*self.qb[1,i]/(2*self.t_sk)
        
        int_qs_tri_02     = 2* ((self.C_a - self.h/2)**2 + (self.h/2)**2)**0.5 / self.t_sk

        # spar 1st half
        int_qb_tri         += (self.qb[0,n_booms_triangle+2]*self.h/2)/(2*self.t_sp)  #line integral of qb/t
        int_qs_tri_01       = - self.h/(2*self.t_sp)
        int_qs_tri_02      += self.h/(2*self.t_sp)

        # spar 2nd half
        int_qb_tri      += (self.qb[0, n_booms_triangle + 2] * self.h / 2) / (2 * self.t_sp)  # line integral of qb/t
        int_qs_tri_01       += - self.h / (2 * self.t_sp) 
        int_qs_tri_02       += self.h / (2 * self.t_sp)

        print(int_qb_circ)
        print(int_qs_circ_01)
        print(int_qs_circ_02)

        print(int_qb_tri)
        print(int_qs_tri_01)
        print(int_qs_tri_02)
                        #qs_1             qs2
        A = np.array([[int_qs_circ_01,int_qs_circ_02],
                      [int_qs_tri_01,int_qs_tri_02]])


        b = np.array([[-int_qb_circ],
                      [-int_qb_tri ]])
        qs01, qs02 =  np.linalg.solve(A,b)
        return qs01, qs02


    def moment_flow_from_shear_mid_spar(self):
        qb               = self.flow_from_shear()
        n_booms_triangle = int((self.n_st - (2 * self.n_arc_half + 1)) / 2)
        n_steps_circ     = (self.n_arc_half*2+1)+1
       ## moment is taken around y=0 at the spar
        tmp = self.dtheta_dx()
        qs01 = tmp[0]
        qs02 = tmp[1]

        qb[0,0:n_booms_triangle+1]                              += qs02
        qb[0,n_booms_triangle]                                  += qs01-qs02
        qb[0,n_booms_triangle+1:n_booms_triangle+n_steps_circ]  += qs01
        qb[0,n_booms_triangle+n_steps_circ+1:n_booms_triangle+n_steps_circ+1+1]  += -qs01+qs02
        qb[0,n_booms_triangle+n_steps_circ+1+1+1:2*n_booms_triangle+n_steps_circ+1+1+1+1]+= qs02

        # cell 1 = semi circle
        n_booms_triangle = int((self.n_st - (2 * self.n_arc_half + 1)) / 2) # number of booms in triangle part (lower or upper)
        print('n_booms_triangle',n_booms_triangle)

        int_start_pos = n_booms_triangle + 2    # start position of shear flow in array
        print('start pos', int_start_pos)
        n_steps_circ = (self.n_arc_half*2+1)+1  #number of steps in semicircle

        q_start = qb[0,int_start_pos-1]+qb[0,int_start_pos-2]    # shear flow at begining of integration
        M_1 = (q_start + qb[0,int_start_pos])/2 *qb[1,int_start_pos]* self.h/2           # integration start
        for i in range(int_start_pos,int_start_pos+n_steps_circ):
            #       ((qbi + qb i+1) * length)/2 * moment arm
            M_1 += (qb[0,i] + qb[0,i + 1])/2 * qb[1,i+1] * self.h/2

        # cell 2 = triangle
        q_start = 0
        int_start_pos = 0
        n_steps_tri = n_booms_triangle + 1
        x = self.boom_loc_area[-self.n_arc_half-1,0] + self.h/2
        y = self.boom_loc_area[-self.n_arc_half-1,1]
        moment_arm = np.sqrt(x**2+y**2)
        M_2 = (q_start + qb[0,int_start_pos])/2 *qb[1,int_start_pos]* moment_arm

        for i in range(int_start_pos,int_start_pos+n_steps_tri):
            M_2 += (qb[0,i] + qb[0,i + 1])/2 * qb[1,i+1] * moment_arm
        M_2 = M_2 * 2                                                    #Due to symmetry

        z_sc = -M_1-M_2+-self.h/2
        print("M1: ", M_1)
        print("M2: ", M_2)
        return z_sc


    def shear_centre(self):



        # # Redundant shear flow in left & right cell
        # # Computed by using matrices and solving for x
        # # q0_1 = x[0], q0_2 = x[1]
        A = np.array([[(1 / (np.pi * h ** 2)) * ((np.pi * h) / (self.G * self.t_sk) +
                                                 (2 * h) / (self.G * self.t_sp)),
                       (-1 / (np.pi * h ** 2)) * (2 * h) / (self.G * self.t_sp)],
                      [((-1) / (2 * h * (self.C_a - h))) * (2 * h) / (self.G * self.t_sp),
                       ((1) / (2 * h * (self.C_a - h))) * ((2 * h) / (self.G * self.t_sp) +
                                                           (2 * L_sk) / (self.G * self.t_sk))]])
        'Old method'
        # b = np.array([[((-1) / (np.pi * h**2)) * ((qb_7 * theta_7 * h) / (self.G * self.t_sk) +
        #                                           (qb_8 * theta * h) / (self.G * self.t_sk) +
        #                                           (qb_9 * theta * h) / (self.G * self.t_sk) +
        #                                           (qb_10 * theta_7 * h) / (self.G * self.t_sk) -
        #                                           (qb_11 * h) / (self.G * self.t_sp) -
        #                                           (qb_6 * h) / (self.G * self.t_sp))],
        #               [((-1) / (2 * h * (self.C_a - h))) * ((qb_12 * s12) / (self.G * self.t_sk) +
        #                         (qb_13 * self.boom_spacing) / (self.G * self.t_sk) +
        #                         (qb_14 * self.boom_spacing) / (self.G * self.t_sk) +
        #                         (qb_15 * self.boom_spacing) / (self.G * self.t_sk) +
        #                         (qb_16 * 0.5 * self.boom_spacing) / (self.G * self.t_sk) +
        #                         (qb_1 * 0.5 * self.boom_spacing) / (self.G * self.t_sk) +
        #                         (qb_2 * self.boom_spacing) / (self.G * self.t_sk) +
        #                         (qb_3 * self.boom_spacing) / (self.G * self.t_sk) +
        #                         (qb_4 * self.boom_spacing) / (self.G * self.t_sk) +
        #                         (qb_5 * s5) / (self.G * self.t_sk) +
        #                         (qb_6 * h) / (self.G * self.t_sp) +
        #                         (qb_11 * h) / (self.G * self.t_sp))]])
        'New method'
        b2 = (self.qb[0, 0] * self.boom_spacing * 0.5 +
              np.sum(self.qb[0, 1:booms_triangle]) * self.boom_spacing +
              s5 * self.qb[0, booms_triangle] +
              s12 * self.qb[0, booms_triangle + 3 + 2 * self.n_arc_half + 2] +
              self.boom_spacing * np.sum(self.qb[0,
                                         booms_triangle + 3 + 2 * self.n_arc_half + 3: 2 * booms_triangle + 3 + 2 * self.n_arc_half + 2]) +
              0.5 * self.boom_spacing * self.qb[0, 2 * booms_triangle + 3 + 2 * self.n_arc_half + 2]) / (
                         self.G * self.t_sk) + (qb_6 * h + qb_11 * h) / (self.G * self.t_sp)

        b1 = - (qb_6 * h + qb_11 * h) / (self.G * self.t_sp) + (abs(theta_7) * h * qb_7 + np.sum(
            self.qb[0, booms_triangle + 3:booms_triangle + 3 + 2 * self.n_arc_half]) * self.boom_spacing + qb_10 * abs(
            theta_7) * h) / (self.G * self.t_sk)
        b = np.array([[b1], [b2]])
        self.x = np.matmul(np.transpose(A), b)
        q0_1 = self.x[0]
        q0_2 = self.x[1]
        # #print(qb_1, qb_2, qb_3, qb_4, qb_5, qb_6, qb_7, qb_8, qb_9, qb_10, qb_11, qb_12, qb_13, qb_14, qb_15, qb_16)

        # add q0 to shear flow

        # adding for top = botom skin
        # lower
        self.qb[0, 0] += q0_2
        self.qb[0, 1:booms_triangle] += q0_2
        self.qb[0, booms_triangle] += q0_2
        # upper
        self.qb[0, booms_triangle + 3 + 2 * self.n_arc_half + 2] += q0_2
        self.qb[0,
        booms_triangle + 3 + 2 * self.n_arc_half + 3: 2 * booms_triangle + 3 + 2 * self.n_arc_half + 2] += q0_2
        self.qb[0, 2 * booms_triangle + 3 + 2 * self.n_arc_half + 2] += q0_2

        # adding for arc

        self.qb[0, booms_triangle + 2] += q0_1
        self.qb[0, booms_triangle + 3:booms_triangle + 3 + 2 * self.n_arc_half] += q0_1
        self.qb[0, booms_triangle + 3 + 2 * self.n_arc_half] += q0_1

        # addaing spars
        self.qb[0, booms_triangle + 1] += q0_2 + q0_1
        self.qb[0, booms_triangle + 3 + 2 * self.n_arc_half + 1] += q0_2 + q0_1

        # # # qs = qb + qs_0
        # qs_1 = qb_1 + q0_2
        # qs_2 = qb_2 + q0_2
        # qs_3 = qb_3 + q0_2
        # qs_4 = qb_4 + q0_2
        # qs_5 = qb_5 + q0_2
        # qs_6 = qb_6 + q0_2 + q0_1
        # qs_7 = qb_7 + q0_1
        # qs_8 = qb_8 + q0_1
        # qs_9 = qb_9 + q0_1
        # qs_10 = qb_10 + q0_1
        # qs_11 = qb_7 + q0_2 + q0_1
        # qs_12 = qb_12 + q0_2
        # qs_13 = qb_13 + q0_2
        # qs_14 = qb_14 + q0_2
        # qs_15 = qb_15 + q0_2
        # qs_16 = qb_16 + q0_2
        # # Shear Centre z and y location (due to symmetry y = 0)
        # self.shear_centre_z = (-1) * ((((qs_7 * theta_7 * h) + (qs_8 * theta * h) + (qs_9 * theta * h) + (qs_10 * theta_7 * h)) * h) +
        #                               (((qs_12 * s12) + (qs_13 * self.boom_spacing) + (qs_14 * self.boom_spacing) + (qs_15 * self.boom_spacing) + (qs_16 * 0.5 * self.boom_spacing)) * np.cos(alpha) * h) +
        #                               (((qs_1 * 0.5 * self.boom_spacing) + (qs_2 * self.boom_spacing) + (qs_3 * self.boom_spacing) + (qs_4 * self.boom_spacing) + (qs_5 * s5)) * np.cos(alpha) * h)) - h

        # self.shear_centre_y = 0
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
        L_sk = math.sqrt((self.C_a - h) ** 2 + h ** 2)
        # Enclosed areas
        A_1 = (np.pi * h**2) / 2
        A_2 = h * (self.C_a * h)
        # Unit torque
        T = 1
        
        # Solve Ax = b with x = q1, q2, dtheta/dx
        A = np.array([[2 * A_1, 2 * A_2, 0],
                      [(1/(2 * A_1)) * ((np.pi * h) / (self.G * self.t_sk) + (2 * h) / (self.G * self.t_sp)), (-1 / (2 * A_1)) * ((2 * h) / (self.G * self.t_sp)), -1], 
                      [(-1/(2 * A_2)) * ((2 * h) / (self.G * self.t_sp)), (1/(2*A_2)) * ((2 * h) / (self.G * self.t_sp) + (2 * L_sk) / (self.G * self.t_sk)), -1]])
        #print(A)
        
        b = np.array([[T], [0], [0]])
        #print(b)
        x = np.linalg.solve(A,b)
        #print(x)
        dtheta_dx = x[2]
        
        self.J = T / (self.G * dtheta_dx)
        #print(self.J)
        
    def plot_aileron(self):
        step_n = 100
        theta_step =np.linspace(-np.pi/2,np.pi/2,step_n)
        self.circ = np.row_stack(([np.cos(theta_step)*self.h/2- self.h/2],[np.sin(theta_step)*self.h/2])) #circ[[z],[y]]
        self.spar = np.row_stack(([np.ones(step_n)* - self.h/2], [np.linspace(- self.h/2,self.h/2,step_n)]))
        self.sk_up = np.row_stack(([np.linspace(-self.h/2,-(self.C_a),step_n)], [np.linspace(self.h/2,0,step_n)]))
        self.sk_down = np.row_stack(([np.linspace(-self.h/2,-(self.C_a),step_n)], [np.linspace(-self.h/2,0,step_n)]))


        plt.plot(self.circ[0,:],self.circ[1,:],'black',label = 'skin')
        plt.plot(self.spar[0,:], self.spar[1,:],'blue',label = 'Spar')
        plt.plot(self.sk_up[0,:], self.sk_up[1,:],'black')
        plt.plot(self.sk_down[0,:], self.sk_down[1,:],'black')
        plt.scatter(self.boom_loc_area[:,0],self.boom_loc_area[:,1],c = 'red',marker = 'D' , label = 'Stiffners')
        plt.title(self.name )
        plt.xlabel('z axes [m]')
        plt.ylabel('y axes [m]')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.grid()
        plt.show()


    
        
            
        
#=======================================================================================
f100 = Aircraft("Fokker 100", 0.505, 1.611, 0.125, 0.498, 1.494, 24.5, 16.1, 1.1, 2.4, 1.2, 1.3, 1.7, 11, 0.389,
                1.245, 30, 49.2)


#====================================================
# Assign all required properties to one term
# Replace 'f100' when analysing a different aircraft
#====================================================
# f100.booms()
# f100.centroid()
# f100.second_moi()
# f100.shear_centre()
# f100.torsional_stiffness()
# #I = [f100.Izz, f100.Iyy, f100.G, f100.J, f100.E, f100.shear_centre_z]
# I = [4.753851442684436e-06, 4.5943507864451845e-05, f100.G, 7.748548555816593e-06, f100.E, -0.08553893540215983] # testing true data
#=======================================================================================

def macaulay(x, x_n, pwr=1):
  "returns result of the step function for [x-x_n]^pwr"
  result = (x-x_n)
  if result>=0:
    return result**pwr
  else:
    return 0

def matrix(alpha, h, x_1, x_2, x_3, x_a, P, d1, d3, I):
    """Constructs the matrix A such that Ax=b for the statically indeterminate
    problem. Where:
    A matrix consists of the LHS of the equations
    x = (R_1y, R_2y, R_3y, R_1z, R_2z, R_3z, R_i, C_1, C_2, C_3, C_4, C_5)
    b = constants = RHS of the equations
    Inputs:
    I = (I_zz, I_yy, G, J, E, z_sc)""" 
    Ky    = (1/(I[4]*I[1]))
    Kz    = (1/(I[4]*I[0]))
    L     = 1/(I[2]*I[3])
    Ksi_1 = x_2-x_a/2
    Ksi_2 = x_2+x_a/2
    z_sc = I[5] # a negative value
    Eta   = - h/2 - z_sc
    mc = macaulay
    alpha = math.radians(alpha) # convert from degrees to radians

    def Alpha(a,b):
    #helper function
        return   ( Kz*np.sin(alpha)/6    *mc(a,b,3)
            - L*Eta*z_sc*np.sin(alpha)   *mc(a,b) 
            - L*Eta*h/2 *np.cos(alpha)   *mc(a,b) )
        
    def Beta(a):
    # Helper function; changing variable will be either x_1, x_2 or x_3
        return (((Kz * np.sin(alpha) / 6) * mc(a, Ksi_2, 3) 
               - L * Eta * z_sc * np.sin(alpha) * mc(a, Ksi_2) 
               - L * Eta * (h/2) * np.cos(alpha) * mc(a, Ksi_2)) * P 
               + Kz * integral_z(5, a)
               - L * Eta * integral_z(3, a, z_sc=z_sc))
                
    def Gamma(a,b):
    #helper function
        return Kz/6 * mc(a, b, 3) + L*(Eta**2)*mc(a, b)
    
    def Delta(a, b):
    # Helper function to make this lengthy expression more readable
         return (Ky * (P * ((np.cos(alpha))**2) / 6) * mc(a, b, 3)
                + L * (h/2)**2 * P * (np.cos(alpha))**2 * mc(a, b) 
                + Kz * P * ((np.sin(alpha))**2) / 6 * mc(a, b, 3) 
                + L * (z_sc)**2 * P * ((np.sin(alpha))**2) * mc(a, b) 
                + 2* z_sc * L * (h/2) * P * np.sin(alpha) * np.cos(alpha) * mc(a, b)
                + np.cos(alpha) * L * (h/2) * integral_z(3, a, z_sc=z_sc) 
                + np.sin(alpha) * z_sc * L * integral_z(3, a, z_sc=z_sc) 
                + Kz * np.sin(alpha) * integral_z(5, a))

    #       x =#(               R_1y,              R_2y,              R_3y,                                 R_1z,                                 R_2z,                                 R_3z,                                  R_i,                   C_1,           C_2,                   C_3,           C_4,                                     C_5)
    A = np.array([[                1,                 1,                 1,                                    0,                                    0,                                    0,                        np.sin(alpha),                     0,             0,                     0,             0,                                       0],#Row 1
                  [                0,                 0,                 0,                                    1,                                    1,                                    1,                        np.cos(alpha),                     0,             0,                     0,             0,                                       0],#Row 2
                  [             -h/2,              -h/2,              -h/2,                                    0,                                    0,                                    0,               -h/2 * (np.cos(alpha)),                     0,             0,                     0,             0,                                       0],#Row 3
                  [                0,                 0,                 0,                                  x_1,                                  x_2,                                  x_3,                  np.cos(alpha)*Ksi_1,                     0,             0,                     0,             0,                                       0],#Row 4
                  [             -x_1,              -x_2,              -x_3,                                    0,                                    0,                                    0,                 -np.sin(alpha)*Ksi_1,                     0,             0,                     0,             0,                                       0],#Row 5
                  [                0,   Gamma(x_1, x_2),   Gamma(x_1, x_3),                                    0,                                    0,                                    0,                    Alpha(x_1, Ksi_1),                   x_1,             1,                     0,             0,                                     Eta],#Row 6
                  [                0,                 0,                 0,                                    0,                 Ky/6*mc(x_1, x_2, 3),                 Ky/6*mc(x_1, x_3, 3), Ky*np.cos(alpha)/6 *mc(x_1, Ksi_1,3),                     0,             0,                   x_1,             1,                                       0],#Row 7
                  [  Gamma(x_2, x_1),                 0,   Gamma(x_2, x_3),                                    0,                                    0,                                    0,                    Alpha(x_2, Ksi_1),                   x_2,             1,                     0,             0,                                     Eta],#Row 8
                  [                0,                 0,                 0,                 Ky/6*mc(x_2, x_1, 3),                                    0,                 Ky/6*mc(x_2, x_3, 3), Ky*np.cos(alpha)/6 *mc(x_2, Ksi_1,3),                     0,             0,                   x_2,             1,                                       0],#Row 9
                  [  Gamma(x_3, x_1),   Gamma(x_3, x_2),                 0,                                    0,                                    0,                                    0,                    Alpha(x_3, Ksi_1),                   x_3,             1,                     0,             0,                                     Eta],#Row 10
                  [                0,                 0,                 0,                 Ky/6*mc(x_3, x_1, 3),                 Ky/6*mc(x_3, x_2, 3),                                    0, Ky*np.cos(alpha)/6 *mc(x_3, Ksi_1,3),                     0,             0,                   x_3,             1,                                       0],#Row 11
                  [Alpha(Ksi_1, x_1), Alpha(Ksi_1, x_2), Alpha(Ksi_1, x_3), Ky*np.cos(alpha)/6 *mc(Ksi_1, x_1,3), Ky*np.cos(alpha)/6 *mc(Ksi_1, x_2,3), Ky*np.cos(alpha)/6 *mc(Ksi_1, x_3,3),                                    0,   Ksi_1*np.sin(alpha), np.sin(alpha),   Ksi_1*np.cos(alpha), np.cos(alpha), -1*z_sc*np.sin(alpha)-h/2*np.cos(alpha)] #Row 12
        ])
    b = np.array([[P*np.sin(alpha)+integral_z(2)],                              #Row 1
                  [P*np.cos(alpha)],                                            #Row 2
                  [-P*np.cos(alpha)*h/2-integral_x(3)],                         #Row 3
                  [P*np.cos(alpha)*Ksi_2],                                      #Row 4
                  [-P*np.sin(alpha)*Ksi_2-integral_z(3)],                       #Row 5
                  [Beta(x_1) + d1 * np.cos(alpha)],                             #Row 6
                  [Ky*np.cos(alpha)/6*mc(x_1,Ksi_2,3)*P-d1*np.sin(alpha)],      #Row 7
                  [Beta(x_2)],                                                  #Row 8
                  [Ky*np.cos(alpha)/6*mc(x_2,Ksi_2,3)*P],                       #Row 9
                  [Beta(x_3) + d3 * np.cos(alpha)],                             #Row 10
                  [Ky*np.cos(alpha)/6*mc(x_3,Ksi_2,3)*P-d3*np.sin(alpha)],      #Row 11
                  [Delta(Ksi_1, Ksi_2)]])                                       #Row 12
    
    condnumber = np.linalg.cond(A)
#    print(condnumber)
#    print(np.linalg.det(A))
    return np.linalg.solve(A, b)

# #=======================================================================================
"Integration functions for z and x direction"

""" This calculates the n'th integral (with minimum of n=1). It is structured so that the program first calculates the definite integral from z=0 till z=C_a= -0.505.
Then, it calculates the indefinite integral along dx. The n'th integral (if n>=2) will than be the definite integral for x=0 till x=l_a=1.611
res is the resolution. Higher value = more accurate, but longer runtime """
def integral_z(n, x_final=1.611, z_sc=None, res=1000):
    # --------------------- input data --------------------------------
    newgrid = copy.deepcopy(grid)
    """ boundaries of the integration """
    x1, x2 = 0, 1.611
    z1, z2 = 0, 0.505

    if z_sc != None:
        aero_data_z = times_z(aero_data, nodes_z, z_sc)
        newgrid = transpose(aero_data_z)

#        coord_sys=(1.611,0,-0.505,0)
#        plt.imshow(transpose(newgrid), extent=coord_sys,interpolation='nearest', cmap=cm.gist_rainbow)
#        plt.colorbar()
#        plt.show()

        # ------------------ main program ---------------------------
    start_time = time.time()  # to calculate runtime of the program

    """ The program can only calculate integrals of functions, not matrixes or wathever.
    This function can only have one variable as input: x-value. It also outputs only one value: y-value (=interpolated aero_data)
    The following defenitinion makes such a function that can later be used in the integral"""

    def cubic_function(x):
        y = cubic_interpolator(matrix, nodes,row, x)
        return y

    """ the function 'spline_coefficient(nodes,row)' converts an array of x-values (=nodes) and an array of y-values (=column of the aero_data) into a matrix. This matrix is necessary to use the function 'spline_interpolator'. (see interpolation file for explenation) """
    nodes = nodes_z
    solution = []
    for row in newgrid:
        matrix = cubic_coefficients(nodes, row)
        """ This calculates the definite integral from z1 to z2 of 'function' """
#        print(cubic_function)
        a = def_integral(cubic_function, z1, z2, res)
#        print(a)
        solution.append(a)
        """ The result is a 1D array of data corresponding to the values of the definite integrals of interpolated columns of the aero_data """

    """ This can be used to check the results for when n=1 """
    if n == 1:
#        x = np.linspace(0, 1.611, len(solution))
#        plt.xlabel('x-axis')
#        plt.plot(x, solution)
#        plt.show()
        return solution

    nodes = nodes_x
    if n == 2:
        row = solution
        matrix = cubic_coefficients(nodes, solution)
        solution = def_integral(cubic_function, x1, x_final, res)

    else:
        for i in range(n - 2):
            row = solution
            matrix = cubic_coefficients(nodes, solution)
            solution = indef_integral(cubic_function, x1, x2, res)
            nodes = np.linspace(x1, x2, len(solution))

        row = solution
        matrix = cubic_coefficients(nodes, solution)
        solution = def_integral(cubic_function, x1, x_final, res)

    end_time = time.time()
    run_time = end_time - start_time  # print run_time to see the time it took the program to compute

#    return solution
    return 0


def integral_x(n, res=1000):
    newgrid = copy.deepcopy(aero_data)
    x1, x2 = 0, 1.611
    z1, z2 = 0, 0.505

    def cubic_function(x):
        y = cubic_interpolator(matrix, nodes, row, x)
        return y

    nodes = nodes_x
    solution = []
    for row in newgrid:
        matrix = cubic_coefficients(nodes, row)
        a = def_integral(cubic_function, x1, x2, res)
        solution.append(a)

    if n == 1:
        x = np.linspace(0, 1.611, len(solution))
        plt.xlabel('z-axis')
        plt.plot(x, solution)
        plt.show()
        return solution

    nodes = nodes_z
    if n == 2:
        row = solution
        matrix = cubic_coefficients(nodes, solution)
        solution = def_integral(cubic_function, z1, z2, res)

    else:
        for i in range(n - 2):
            row = solution
            matrix = cubic_coefficients(nodes, solution)
            solution = indef_integral(cubic_function, z1, z2, res)
            nodes = np.linspace(z1, z2, len(solution))

        row = solution
        matrix = cubic_coefficients(nodes, solution)
        solution = def_integral(cubic_function, z1, z2, res)

#    return solution   
    return 0

#=====================================================================
""""Deformation plotting"""


def deflectionplot(I,unknowns,plot, whichplot = None , totalnodes=20):
    """
    plot = "deflection_y" or "deflection_z" or "twist"
    whichplot = "1" or "2" or "3" or "4" for specific plot to save time
    """
    R1y = unknowns[0][0]
    R2y = unknowns[1][0]
    R3y = unknowns[2][0]
    R1z = unknowns[3][0]
    R2z = unknowns[4][0]
    R3z = unknowns[5][0]
    Ri  = unknowns[6][0]
    C1  = unknowns[7][0]
    C2  = unknowns[8][0]
    C3  = unknowns[9][0]
    C4  = unknowns[10][0]
    C5  = unknowns[11][0]
    P   = f100.P
    h   = f100.h
    x1  = f100.x_1
    x2  = f100.x_2
    x3  = f100.x_3
    xa  = f100.x_a
    ksi1    = x2 - xa/2
    ksi2    = x2 + xa/2
    alpha   = math.radians(f100.theta)
    l_a     = f100.l_a
    """I = (I_zz, I_yy, G, J, E, z_sc)""" 
    zsc = I[5]
    eta = -h/2 - zsc
    L = 1/(I[2]*I[3])
    Kz = 1/ (I[4] * I[0])
    Ky = 1/ (I[4] * I[1])
    
    """ functions """
    def v_deflection(x): # plot 1
        v = ( Kz* (R1y/6*macaulay(x,x1,3) 
            + R2y/6*macaulay(x,x2,3) 
            + R3y/6*macaulay(x,x3,3) 
            + Ri*np.sin(alpha)/6*macaulay(x,ksi1,3) 
            - P*np.sin(alpha)/6*macaulay(x,ksi2,3) 
            - integral_z(5,x)) 
            + C1*x 
            + C2 )
        return v
    
    def dvdx(x): # plot 2
        dvdx = ( Kz* (R1y/2*macaulay(x,x1,2) 
            + R2y/2*macaulay(x,x2,2) 
            + R3y/2*macaulay(x,x3,2) 
            + Ri*np.sin(alpha)/2*macaulay(x,ksi1,2) 
            - P*np.sin(alpha)/2*macaulay(x,ksi2,2) 
            - integral_z(4,x)) 
            + C1)
        return dvdx
    
    def Mz(x): # plot 3
        Mz = -1 *(R1y*macaulay(x,x1,1) 
            + R2y*macaulay(x,x2,1) 
            + R3y*macaulay(x,x3,1) 
            + Ri*np.sin(alpha)*macaulay(x,ksi1,1) 
            - P*np.sin(alpha)*macaulay(x,ksi2,1) 
            - integral_z(3,x))
        print(Mz)
        return Mz
    
    def Sy(x): # plot 4
        Sy = -1 *(R1y*macaulay(x,x1,0) 
            + R2y*macaulay(x,x2,0) 
            + R3y*macaulay(x,x3,0) 
            + Ri*np.sin(alpha)*macaulay(x,ksi1,0) 
            - P*np.sin(alpha)*macaulay(x,ksi2,0) 
            - integral_z(2,x))
        return Sy
    
    def w_deflection(x): # plot 1
        w = ( Ky*( R1z/6* macaulay(x,x1,3) 
            + R2z/6*macaulay(x,x2,3) 
            + R3z/6*macaulay(x,x3,3) 
            + Ri*np.cos(alpha)/6*macaulay(x,ksi1,3) 
            - P*np.cos(alpha)/6*macaulay(x,ksi2,3)) 
            + C3*x 
            + C4 )
        return w
    
    def dwdx(x): # plot 2
        dwdx = (Ky*( R1z/2* macaulay(x,x1,2) 
            + R2z/2*macaulay(x,x2,2) 
            + R3z/2*macaulay(x,x3,2) 
            + Ri*np.cos(alpha)/2*macaulay(x,ksi1,2) 
            - P*np.cos(alpha)/2*macaulay(x,ksi2,2)) 
            + C3)
        return dwdx
    
    def My(x): # plot 3
        My = -1* ( R1z* macaulay(x,x1,1) 
            + R2z*macaulay(x,x2,1) 
            + R3z*macaulay(x,x3,1) 
            + Ri*np.cos(alpha)*macaulay(x,ksi1,1) 
            - P*np.cos(alpha)*macaulay(x,ksi2,1))
        print(My)
        return My
    
    def Sz(x): # plot 4
        Sz = -1* ( R1z* macaulay(x,x1,0) 
            + R2z*macaulay(x,x2,0) 
            + R3z*macaulay(x,x3,0) 
            + Ri*np.cos(alpha)*macaulay(x,ksi1,0) 
            - P*np.cos(alpha)*macaulay(x,ksi2,0))
        return Sz
    
    def twist(x): # plot 1
        twist = 1*( L* ( eta*R1y*macaulay(x,x1,1) 
        + eta*R2y*macaulay(x,x2,1) 
        + eta*R3y*macaulay(x,x3,1) 
        - zsc*Ri*np.sin(alpha)*macaulay(x,ksi1,1) 
        - h/2*Ri*np.cos(alpha)*macaulay(x,ksi1,1) 
        + zsc*P*np.sin(alpha)*macaulay(x,ksi2,1) 
        + h/2*P*np.cos(alpha)*macaulay(x,ksi2,1) 
        - integral_z(3,x_final=x,z_sc=zsc) ) 
        + C5 ) 
        print(twist)
        return twist
    
    def torque(x): # plot 2
        torque = 1*( eta*R1y*macaulay(x,x1,0) 
        + eta*R2y*macaulay(x,x2,0) 
        + eta*R3y*macaulay(x,x3,0) 
        - zsc*Ri*np.sin(alpha)*macaulay(x,ksi1,0) 
        - h/2*Ri*np.cos(alpha)*macaulay(x,ksi1,0) 
        + zsc*P*np.sin(alpha)*macaulay(x,ksi2,0) 
        + h/2*P*np.cos(alpha)*macaulay(x,ksi2,0) 
        + integral_z(2,x_final=x,z_sc=zsc) ) 
        print(torque)
        return torque
        
    def dist_torque(x): # plot 3
#        nodes = np.linspace(0, 1.611, 41)
#        matrix = spline_coefficient(nodes, integral_z(1,x,z_sc=zsc))
        return integral_z(1,x,zsc)
    
    if plot == "deflection_y":
        if whichplot == None:
            ydata_1 = []
            ydata_2 = []
            ydata_3 = []
            ydata_4 = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata_1 = v_deflection(i)
                ydata_1.append(ptydata_1)
                ptydata_2 = dvdx(i)
                ydata_2.append(ptydata_2)
                ptydata_3 = Mz(i)
                ydata_3.append(ptydata_3)
                ptydata_4 = Sy(i)
                ydata_4.append(ptydata_4)
                
            plt.figure()     
            plot1 = plt.subplot(2, 2, 1)
            plot1.set_title("Deflection in y")
            plot1.set_xlabel('x')
            plot1.set_ylabel('v(x)')
            plt.plot(xdata, ydata_1,label='Num. model',color='orange')   
            plt.legend()
            plot2 = plt.subplot(2, 2, 2)
            plot2.set_title("Slope in y")
            plot2.set_xlabel('x')
            plot2.set_ylabel('dv/dx(x)')
            plt.plot(xdata, ydata_2,label='Num. model',color='orange')   
            plt.legend()
            plot3 = plt.subplot(2, 2, 3)
            plot3.set_title("Bending moment about z")
            plot3.set_xlabel('x')
            plot3.set_ylabel('M\z(x)')
            plt.plot(xdata, ydata_3,label='Num. model',color='orange') 
            plt.legend()
            plot4 = plt.subplot(2, 2, 4)
            plot4.set_title("Shear force in y")
            plot4.set_xlabel('x')
            plot4.set_ylabel('S\y(x)')
            plt.plot(xdata, ydata_4,label='Num. model',color='orange')
            plt.legend()
            plt.tight_layout()
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()
        if whichplot == 1:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = v_deflection(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()
        elif whichplot == 2:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = dvdx(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()  
        elif whichplot == 3:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = Mz(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()  
        elif whichplot == 4:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = Sy(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()  
            
    elif plot == "deflection_z":
        if whichplot == None:
            ydata_1 = []                                                                        
            ydata_2 = []
            ydata_3 = []
            ydata_4 = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata_1 = w_deflection(i)
                ydata_1.append(ptydata_1)
                ptydata_2 = dwdx(i)
                ydata_2.append(ptydata_2)
                ptydata_3 = My(i)
                ydata_3.append(ptydata_3)
                ptydata_4 = Sz(i)
                ydata_4.append(ptydata_4)
                
                
            plt.figure()     
            plot1 = plt.subplot(2, 2, 1)
            plot1.set_title("Deflection in z")
            plot1.set_xlabel('x')
            plot1.set_ylabel('w(x)')
            plt.plot(xdata, ydata_1,label='Num. model',color='orange')     
            plt.legend()
            plot2 = plt.subplot(2, 2, 2)
            plot2.set_title("Slope in z")
            plot2.set_xlabel('x')
            plot2.set_ylabel('dw/dx(x)')
            plt.plot(xdata, ydata_2,label='Num. model',color='orange')    
            plt.legend()
            plot3 = plt.subplot(2, 2, 3)
            plot3.set_title("Bending moment about y")
            plot3.set_xlabel('x')
            plot3.set_ylabel('M\y(x)')
            plt.plot(xdata, ydata_3,label='Num. model',color='orange')
            plt.legend()
            plot4 = plt.subplot(2, 2, 4)
            plot4.set_title("Shear force in z")
            plot4.set_xlabel('x')
            plot4.set_ylabel('S\z(x)')
            plt.plot(xdata, ydata_4,label='Num. model',color='orange')
            plt.legend()
            plt.tight_layout()
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()
        
        if whichplot == 1:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = w_deflection(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()
        elif whichplot == 2:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = dwdx(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()  
        elif whichplot == 3:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = My(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()  
        elif whichplot == 4:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = Sz(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()  
            
    elif plot == "twist":
        if whichplot == None:
            ydata_1 = []    
            ydata_2 = []
            ydata_3 = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata_1 = twist(i)
                ydata_1.append(ptydata_1)
                ptydata_2 = torque(i)
                ydata_2.append(ptydata_2)
                
            plt.figure()     
            plot1 = plt.subplot(2, 2, 1)
            plot1.set_title("Twist")
            plot1.set_xlabel('x')
            plot1.set_ylabel('θ(x)')
            plt.plot(xdata, ydata_1,label='Num. model',color='orange')  
            plt.legend()
            plot2 = plt.subplot(2, 2, 2)
            plot2.set_title("Torque")
            plot2.set_xlabel('x')
            plot2.set_ylabel('T(x)')
            plt.plot(xdata, ydata_2,label='Num. model',color='orange')   
            plt.legend()
            plot3 = plt.subplot(2, 2, 3)
            plot3.set_title("Distributed torque")
            plot3.set_xlabel('x')
            plot3.set_ylabel('τ(x)')
            plt.plot(np.linspace(0,f100.l_a,41),integral_z(1),label='Num. model',color='orange')    
            plt.legend()
            plt.tight_layout()
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()
        if whichplot == 1:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = twist(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()
        elif whichplot == 2:
            ydata = []
            xdata = np.linspace(0,l_a,totalnodes)
            for i in xdata:
                ptydata = torque(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()  
        elif whichplot == 3:
#            ydata = integral_z(1,zsc)
            ydata = []
            xdata = np.linspace(0,l_a,41)
            for i in xdata:
                ptydata = dist_torque(i)
                ydata.append(ptydata)
            plt.figure()   
            plt.plot(xdata, ydata)
            plt.show()  
        elif whichplot == 4:
            print("no plot 4 for twist")



