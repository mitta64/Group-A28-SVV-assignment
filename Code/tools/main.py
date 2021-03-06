import math
import numpy as np
import matplotlib.pyplot as plt
from tools import *
from Grid import Grid

# =======================================================================================
f100 = Aircraft("Fokker 100", 0.505, 1.611, 0.125, 0.498, 1.494, 24.5, 16.1, 1.1, 2.4, 1.2, 1.3, 1.7, 11, 0.389, 1.245,
                30, 49.2)
A320 = Aircraft("Airbus A320", 0.547, 2.771, 0.153, 1.281, 2.681, 28., 22.5, 1.1, 2.9, 1.2, 1.5, 2., 17, 1.103, 1.642,
                26., 91.7)

# ====================================================
# Assign all required properties to one term
# Replace 'f100' when analysing a different aircraft

# ====================================================

#I = [f100.Izz, f100.Iyy, f100.G, f100.J, f100.E, f100.shear_centre_z]
#=======================================================================================
print("[f100.Izz, f100.Iyy, f100.G, f100.J, f100.E, f100.shear_centre_z]")
I_1 = [f100.Izz, f100.Iyy, f100.G, 0, f100.E, 0]
I_2 = [4.753851442684436e-06, 4.5943507864451845e-05, f100.G, 7.748548555816593e-06, f100.E,
     -0.08553893540215983]  # testing true data
print(I_1)
print(I_2)
print(np.array(I_1)-np.array(I_2))

# =======================================================================================
# print([f100.Izz, f100.Iyy, f100.G, f100.J, f100.E, f100.shear_centre_z])

# unknowns = matrix(f100.theta, f100.h, f100.x_1, f100.x_2, f100.x_3, f100.x_a, f100.P, f100.d_1, f100.d_3, I)



# v_def = []
# X = np.linspace(0,1,10)
# for x in X:
# 	v_def.append(v_deflection(x, unknowns,f100))
# plt.plot(X,v_def)#,'v_deflection', 'm')
# plt.show()


#======
"""
For validating the structurally indeterminate problem solver we will use some
structural values found by the verification model for the B737. Specifically 
the values pertaining to shear.
"""
cross_ver = {'nst': 15, 'Ca': 0.605, 'ha': 0.205, 'tsk': 0.0011, 
			 'tsp': 0.0028, 'tst': 0.0012, 'hst': 0.016, 'wst': 0.019, 
			 'totarea': 0.002686478946739162, 'zc': -0.24048766835061938, 
			 'yc': -4.339264140786877e-19, 'Izz': 1.0280189203385745e-05, 
			 'Iyy': 8.651211860639685e-05, 'zsc': -0.10856995078063854, 
			 'ysc': 0, 'J': 1.5101498390705797e-05, 'lsk': 0.512847443203142, 
			 'lcirc': 0.32201324699295375, 'P': 1.3477081333992378, 
			 'spacing': 0.08984720889328252, 'ncirc': 3.0, 'nsk': 12.0, 
			 'stcoord': np.array([[-0.        ,  0.        ],
						       	  [-0.03692049,  0.07877549],
						       	  [-0.12081074,  0.09876497],
						       	  [-0.20884515,  0.08080771],
						       	  [-0.29687956,  0.06285044],
						       	  [-0.38491397,  0.04489317],
						       	  [-0.47294838,  0.0269359 ],
						       	  [-0.56098279,  0.00897863],
						       	  [-0.56098279, -0.00897863],
						       	  [-0.47294838, -0.0269359 ],
						       	  [-0.38491397, -0.04489317],
						       	  [-0.29687956, -0.06285044],
						       	  [-0.20884515, -0.08080771],
						       	  [-0.12081074, -0.09876497],
						       	  [-0.03692049, -0.07877549]]), 
			 'phi': 0.8765581355442198, 'areast': 4.2e-05, 'zsp': -0.1025, 
			 'ysp': 0, 'areasp': 0.000574, 'Izzsp': 2.0101958333333327e-06, 
			 'Iyysp': 0, 'zsk': -0.35375, 'ysk': 0.05125, 
			 'areask': 0.0005641321875234562, 'Izzsk': 4.939094829306926e-07, 
			 'Iyysk': 1.1870575264653724e-05, 'zcirc': -0.03724647333232291, 
			 'ycirc': 0, 'areacirc': 0.00035421457169224914, 
			 'Izzcirc': 1.8607334219208462e-06, 
			 'Iyycirc': 3.524797199058249e-07, 
			 'Qz': -0.0006460650579743288, 
			 'Qy': -1.1657341758564143e-21}

			     #name,  C_a,  l_a,  x_1,  x_2,  x_3, x_a,    h, t_sk,  t_sp,t_st, h_st, w_st,n_st,  d_1,  d_3,theta,P)
B737 = Aircraft("B737",0.605,2.661,0.172,1.211,2.591, 35., 20.5,  1.1,   2.8, 1.2,  1.6,  1.9,  15,1.154,1.840,   28.,97.4)

#I = (   I_zz,     I_yy,      G,   			  J, E,     z_sc)
I = [B737.Izz, B737.Iyy, B737.G, cross_ver['J'], B737.E, cross_ver['zsc']]

unknowns = matrix(B737.theta,
				  B737.h,
				  B737.x_1,
				  B737.x_2,
				  B737.x_3,
				  B737.x_a,
				  B737.P,
				  B737.d_1,
				  B737.d_3,
				  I)

xcoord = np.arange(0, B737.l_a, 24.5714283/1000)

deflection_z = deflectionplot(B737, I,unknowns,"w",xcoord= xcoord)
deflection_y = deflectionplot(B737, I,unknowns,"v",xcoord= xcoord)

for i,j in enumerate("(R_1y, R_2y, R_3y, R_1z, R_2z, R_3z, R_i, C_1, C_2, C_3, C_4, C_5)".split(',')):
	print(j, unknowns[i])

