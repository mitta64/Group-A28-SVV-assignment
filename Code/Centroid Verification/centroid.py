import math
import numpy as np
import matplotlib.pyplot as plt
import time
from data import aero_data, grid, transpose
import copy

class TestClass(object):
    def __init__(self, name, C_a, l_a, x_1, x_2, x_3, x_a, h,
                 t_sk, t_sp, t_st, h_st, w_st, n_st, d_1, d_3, theta, P):
        self.name   = name
        self.C_a    = C_a                   # "Chord length aileron[m]"
        self.l_a    = l_a                   # "Span of the aileron[m]"
        self.x_1    = x_1                   # "x-location of hinge 1 [m]"
        self.x_2    = x_2                   # "x-location of hinge 2 [m]"
        self.x_3    = x_3                   # "x-location of hinge 3 [m]"
        self.x_a    = round(x_a / 100, 8)   # "Distance between actuator 1 and 2 [m]"
        self.h      = round(h / 100, 10)     # "Aileron height[m]"
        self.t_sk   = round(t_sk / 1000, 8) # "Skin thickness [m]"
        self.t_sp   = round(t_sp / 1000, 8) # "Spar thickness [m]"
        self.t_st   = round(t_st / 1000, 8) # "Thickness of stiffener[m]"
        self.h_st   = round(h_st / 100, 8)  # "Height of stiffener[m]"
        self.w_st   = round(w_st / 100, 8)  # "Width of stiffener[m]"
        self.n_st   = n_st                  # "Number of stiffeners [-]"
        self.d_1    = round(d_1 / 100, 8)   # "Vertical displacement hinge 1[m]"
        self.d_3    = round(d_3 / 100, 8)   # "Vertical displacement hinge 3[m]"
        self.theta  = theta                 # "Maximum upward deflection[deg]"
        self.P      = round(P * 1000, 8)    # "Load in actuator 2[N]"

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
        angle_triangle = (math.atan((self.h /2) / (self.C_a - (self.h /2))))                                                
        
        # Start array with Col 1 z coordinate & Col 2 y coordinate 
        # Add stringers, starting at LE and going clockwise
        self.boom_loc_area = np.zeros(shape=(self.n_st, 3))
        # calc amount of stringers in arc
        self.n_arc_half = int((np.pi / 2) / self.angle_arc)
        #print('n_arc', self.n_arc_half)
        # Add stringer in 0,0 arc
        self.boom_loc_area[0,:] = np.array([0,0,self.boom_area])

        # Add stringer in upper arc section
        for i in np.arange(1, self.n_arc_half+1, 1, dtype=int):
            boom_arc_up = np.array([-((self.h / 2) - (np.cos(self.angle_arc * i) * (self.h / 2))),
                                    np.sin(self.angle_arc * i) * (self.h / 2), self.boom_area])
            self.boom_loc_area[i, :] = boom_arc_up

            # boom_arc_down = (np.array([
            # -((self.h / 2) - (np.cos(self.angle_arc) * (self.h / 2))),
            # - np.sin(self.angle_arc) * (self.h / 2), self.boom_area]))

            # self.boom_loc_area[i+1,:] = boom_arc_down
            #print('upper arc',i, boom_arc_up)

        # Add stringers in upper triangular section

        pos = self.n_arc_half+1
        for i in np.arange((self.n_st - (self.n_arc_half * 2 + 1)) / 2 - 0.5, -0.5, -1):
            boom_tri_up = np.array([-(self.C_a - i * self.boom_spacing * np.cos(angle_triangle)),
                                    i * self.boom_spacing * np.sin(angle_triangle), self.boom_area])
            self.boom_loc_area[pos, :] = boom_tri_up

            #print('upper tri',pos,boom_tri_up,i)
            pos = pos + 1

        # Add stringers in lower triangular section
        for i in np.arange(0.5, (self.n_st - (self.n_arc_half * 2 + 1)) / 2 + 0.5, 1):
            boom_tri_down = np.array([-(self.C_a - i * self.boom_spacing * np.cos(angle_triangle)),
                                      -i * self.boom_spacing * np.sin(angle_triangle), self.boom_area])
            self.boom_loc_area[pos] = boom_tri_down

            #print('lower tri',pos,boom_tri_down,i)
            pos = pos + 1

        # Add in lower arc section

        for i in np.arange(self.n_arc_half,0,-1 ,dtype=int):
            boom_arc_down = (np.array([
                -((self.h / 2) - (np.cos(self.angle_arc *  (i)) * (self.h / 2))),
                - np.sin(self.angle_arc * (i)) * (self.h / 2), self.boom_area]))

            self.boom_loc_area[pos, :] = boom_arc_down
            #print('lower arc',pos,boom_arc_down)
            pos = pos + 1

            # "Final output of booms function is self.boom_loc_area"

    def centroid(self):
            arr_z_y_a = np.zeros(shape=(3, 4 + self.n_st))

            x_circ = - (self.h / 2 - self.h / np.pi)
            a_circ = np.pi * self.h / 2 * self.t_sk
            arr_z_y_a[:, 0] = [x_circ, 0., a_circ]

            x_spr = - self.h / 2
            a_spr = self.h * self.t_sp
            arr_z_y_a[:, 1] = [x_spr, 0., a_spr]

            x_sk = - (self.C_a / 2 + self.h / 4 )
            y_sk = self.h / 4
            a_sk = np.sqrt((self.h / 2) ** 2 + (self.C_a - self.h / 2) ** 2) * self.t_sk
            arr_z_y_a[:, 2:4] = [[x_sk, x_sk], [y_sk, -y_sk], [a_sk, a_sk]]

            arr_z_y_a[:, 4:] = np.transpose(self.boom_loc_area)
            self.boom_skin_z_y_a = arr_z_y_a

            self.cent = np.round(np.array([[np.sum(arr_z_y_a[0, :] * arr_z_y_a[2, :]) / np.sum([arr_z_y_a[2, :]])],
                                           [np.sum(arr_z_y_a[1, :] * arr_z_y_a[2, :]) / np.sum([arr_z_y_a[2, :]])]]), 9)


















f100 = TestClass("Fokker 100", 
                    1.,    # "Chord length aileron[m]"
                    1.611,  # "Span of the aileron[m]
                    0.125,  # "x-location of hinge 1 [m
                    0.498,  # "x-location of hinge 2 [m 
                    1.494,  # "x-location of hinge 3 [m
                    24.5,   # "Distance between actuator 1 and 2
                    10.,      # "Aileron height[m]"
                    1.,      # "Skin thickness [m]"
                    1.,      # "Spar thickness [m]"
                    1.,      # "Thickness of stiffener[m]"
                    1.,      # "Height of stiffener[m]"
                    1.,      # "Width of stiffener[m]"
                    1,     # "Number of stiffeners [-]"
                    0.389,  # "Vertical displacement hinge 1[m]
                    1.245,  # "Vertical displacement hinge 3[m]
                    30,     # "Maximum upward deflection[deg]"
                    49.2)   # "Load in actuator 2[N]

print(f100.h)
f100.booms()
f100.centroid()
print(f100.cent)
print(f100.boom_loc_area)
plt.scatter(f100.boom_loc_area[:,0]*1000,f100.boom_loc_area[:,1]*1000)
plt.gca().invert_xaxis()
plt.show()

#results of manual calc:
     #    Arc    Spar      Plate
Y = [ 0.0185,   0.05,     0.575,     0.575]
A = [0.00316, 0.0001, 0.0009513, 0.0009513]


cy = 0
for i in range(len(Y)):
    cy+=(Y[i]*A[i])

cy+= sum((f100.boom_loc_area[:,0]*-1)*f100.boom_loc_area[:,2])
cy/=(sum(A)+sum(f100.boom_loc_area[:,2]))
print 1-cy