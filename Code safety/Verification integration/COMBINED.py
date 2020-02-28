# applying all functions to solve integrals
import numpy as np
import matplotlib.pyplot as plt
import time

from data import aero_data, grid, f100, transpose
from integrator import def_integral, indef_integral
from interpolator import spline_coefficient, spline_interpolator, cubic_coefficient, cubic_interpolator


""" This calculates the n'th integral (with minimum of n=1). It is structured so that the program first calculates the definite integral from z=0 till z=C_a= -0.505.
Then, it calculates the indeffinite integral along dx. The n'th integral (if n>=2) will than be the definite integral for x=0 till x=l_a=1.611
res is the resolution. Higher value = more accurate, but longer runtime """
def integral_z(n,x_final=1.611,z_sc=0,res=1000):
    #--------------------- input data --------------------------------
    """ boundaries of the integration """
    x1 ,x2 = 0, 1.611
    z1, z2 = 0, 0.505

    for row in range(len(grid)):
        for element in range(len(row)):
            z = element*0.505/80
            grid[row][element] = grid[row][element]*(z-z_sc)
        
    
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

