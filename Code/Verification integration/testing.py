# applying all functions to solve integrals
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm

from data import transpose
from integrator import def_integral, indef_integral
from interpolator import spline_coefficient, spline_interpolator, cubic_coefficients, cubic_interpolator
import copy

#--------------------- grid of input function --------------------------------

test_grid = []
for z in range(81):
    z=z*15/80
    row=[]
    for x in range(41):
        x=x*5/40
        w = (1+x+x*x)*z
        row.append(w)
    test_grid.append(row)




def integral_x(n, res=1000):
    newgrid = copy.deepcopy(test_grid)
    x1, x2 = 0, 5
    z1, z2 = 0, 15

    def cubic_function(x):
        y = cubic_interpolator(matrix, nodes, row, x)
        return y

    nodes = np.linspace(0,5,41)
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

    nodes = np.linspace(0,15,81)
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

    return solution
