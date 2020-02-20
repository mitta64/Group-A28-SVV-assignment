# applying all functions to solve integrals
import numpy as np
import matplotlib.pyplot as plt
import time

from data import aero_data, grid, f100, transpose
from integrator import def_integral, indef_integral
from interpolator import spline_coefficient, spline_interpolator

#--------------------- input data --------------------------------
x1 ,x2 = 0, 1.611
y1, y2 = 0, 0.505
n = 2   # number of integral you want
res = 1000


#------------------ main program ---------------------------
resolution = [5000]
for res in resolution:
    start_time = time.time()

    def function(x):
        return spline_interpolator(matrix, nodes, x)

    nodes = np.linspace(y1,y2,len(grid))
    solution = []
    for row in grid:
        matrix = spline_coefficient(nodes, row)
        a = def_integral(function,y1,y2,res)
        solution.append(a)

    for i in range(n-1):
        nodes = np.linspace(x1,x2,len(solution))
        matrix = spline_coefficient(nodes, solution)
        solution = indef_integral(function,x1,x2,res)

    #solution = def_integral(function,x1,x2,res)

    end_time = time.time()
    print(res, end_time - start_time, solution)

    ##x = np.linspace(0,1.611,len(solution_n))
    ##plt.xlabel('x-axis')
    ##plt.ylabel('z-axis')
    ##plt.plot(x,solution_n)
    ##plt.show()


