import math
import numpy as np
import matplotlib.pyplot as plt
from tools import *

# =======================================================================================
f100 = Aircraft("Fokker 100", 0.505, 1.611, 0.125, 0.498, 1.494, 24.5, 16.1, 1.1, 2.4, 1.2, 1.3, 1.7, 11, 0.389, 1.245,
                30, 49.2)
A320 = Aircraft("Airbus A320", 0.547, 2.771, 0.153, 1.281, 2.681, 28., 22.5, 1.1, 2.9, 1.2, 1.5, 2., 17, 1.103, 1.642,
                26., 91.7)

# ====================================================
# Assign all required properties to one term
# Replace 'f100' when analysing a different aircraft

# ====================================================
f100.booms()
f100.centroid()
f100.second_moi()
f100.shear_centre()
f100.torsional_stiffness()
#I = [f100.Izz, f100.Iyy, f100.G, f100.J, f100.E, f100.shear_centre_z]
#=======================================================================================
# I = [f100.Izz, f100.Iyy, f100.G, f100.J, f100.E, f100.shear_centre_z]
I = [4.753851442684436e-06, 4.5943507864451845e-05, f100.G, 7.748548555816593e-06, f100.E,
     -0.08553893540215983]  # testing true data
# =======================================================================================
unknowns = matrix(f100.theta, f100.h, f100.x_1, f100.x_2, f100.x_3, f100.x_a, f100.P, f100.d_1, f100.d_3, I)

v_def = []
X = np.linspace(0,1,10)
for x in X:
	v_def.append(v_deflection(x, unknowns,f100))
plt.plot(X,v_def)#,'v_deflection', 'm')
plt.show()