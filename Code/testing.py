# applying all functions to solve integrals
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm

from data import aero_data, grid, f100, transpose
from integrator import def_integral, indef_integral
from interpolator import spline_coefficient, spline_interpolator

#--------------------- input data --------------------------------

test_grid = []
for z in range(81):
    z=z*15/81
    row=[]
    for x in range(41):
        x=x*5/41
        w = (1+x+x*x)*z
        row.append(w)
    test_grid.append(row)
grid=transpose(test_grid)

x1,x2 = 0,5
z1,z2 = 0,15

#------------------ main program ---------------------------
res = 1000

def function(x):
    return spline_interpolator(matrix, nodes, x)

nodes = np.linspace(z1,z2,len(grid))
solution = []
for row in grid:
    matrix = spline_coefficient(nodes, row)
    a = def_integral(function,z1,z2,res)
    solution.append(a)

print(solution)

nodes = np.linspace(x1,x2,len(solution))
matrix = spline_coefficient(nodes, solution)
solution = def_integral(function,x1,x2,res)


coord_sys=(0,5,0,15)
plt.imshow(test_grid, extent=coord_sys,interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()

##x = np.linspace(0,5,len(solution))
##plt.xlabel('x-axis')
##plt.ylabel('z-axis')
##plt.plot(x,solution)
##plt.show()


