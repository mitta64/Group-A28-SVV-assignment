# applying all functions to solve integrals
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

from data import aero_data, grid, f100, transpose, nodes_z, nodes_x, times_z
from integrator import def_integral, indef_integral
from interpolator import spline_coefficient, spline_interpolator


""" This calculates the n'th integral (with minimum of n=1). It is structured so that the program first calculates the definite integral from z=0 till z=C_a= -0.505.
Then, it calculates the indefinite integral along dx. The n'th integral (if n>=2) will than be the definite integral for x=0 till x=l_a=1.611
res is the resolution. Higher value = more accurate, but longer runtime """
def integral_z(n,x_final=1.611,z_sc=None,res=1000):
    #--------------------- input data --------------------------------
    newgrid = copy.deepcopy(grid)
    """ boundaries of the integration """
    x1 ,x2 = 0, 1.611
    z1, z2 = 0, 0.505
    
    if z_sc != None:
        newgrid = times_z(aero_data, nodes_z, z_sc)        
    
    #------------------ main program ---------------------------
    start_time = time.time() # to calculate runtime of the program

    """ The program can only calculate integrals of functions, not matrixes or wathever.
    This function can only have one variable as input: x-value. It also outputs only one value: y-value (=interpolated aero_data)
    The following defenitinion makes such a function that can later be used in the integral"""
    def spline_function(x):
        y = spline_interpolator(matrix, nodes, x)
        return y

    """ the function 'spline_coefficient(nodes,row)' converts an array of x-values (=nodes) and an array of y-values (=column of the aero_data) into a matrix. This matrix is necessary to use the function 'spline_interpolator'. (see interpolation file for explenation) """
    nodes = nodes_z
    solution = []
    for row in newgrid:
        matrix = spline_coefficient(nodes, row)
        """ This calculates the definite integral from z1 to z2 of 'function' """
        a = def_integral(spline_function,z1,z2,res)
        solution.append(a)
        """ The result is a 1D array of data corresponding to the values of the definite integrals of interpolated columns of the aero_data """

    """ This can be used to check the results for when n=1 """
    if n == 1:
        x = np.linspace(0,1.611,len(solution))
        plt.xlabel('x-axis')
        plt.plot(x,solution)
        plt.show()
        return solution
    
    nodes = nodes_x
    if n == 2:
        matrix = spline_coefficient(nodes, solution)
        solution = def_integral(spline_function,x1,x2,res)        

    else:
        for i in range(n-2):
            matrix = spline_coefficient(nodes, solution)
            solution = indef_integral(spline_function,x1,x2,res)
            nodes = np.linspace(x1,x2,len(solution))

        matrix = spline_coefficient(nodes, solution)
        solution = def_integral(spline_function,x1,x2,res)

    return solution
        
    end_time = time.time()
    run_time = end_time - start_time   # print run_time to see the time it took the program to compute

    return solution




def integral_x(n,res=1000):
    newgrid = copy.deepcopy(aero_data)
    x1 ,x2 = 0, 1.611
    z1, z2 = 0, 0.505

    def spline_function(x):
        y = spline_interpolator(matrix, nodes, x)
        return y

    nodes = nodes_x
    solution = []
    for row in newgrid:
        matrix = spline_coefficient(nodes, row)
        a = def_integral(spline_function,x1,x2,res)
        solution.append(a)

    if n == 1:
        x = np.linspace(0,1.611,len(solution))
        plt.xlabel('x-axis')
        plt.ylabel('z-axis')
        plt.plot(x,solution)
        plt.show()
        return solution
        
    nodes = nodes_z
    if n == 2:
        matrix = spline_coefficient(nodes, solution)
        solution = def_integral(spline_function,z1,z2,res)        

    else:
        for i in range(n-2):
            matrix = spline_coefficient(nodes, solution)
            solution = indef_integral(spline_function,z1,z2,res)
            nodes = np.linspace(z1,z2,len(solution))

        matrix = spline_coefficient(nodes, solution)
        solution = def_integral(spline_function,z1,z2,res)

    return solution
