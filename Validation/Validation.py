import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#Get nodes
Displ_coordinates          = np.genfromtxt("B737.inp",skip_header = 9,skip_footer=7996,delimiter=',')
Boundarynode_coordinates  = np.genfromtxt("B737.inp",skip_header = 14146, comments = "*",skip_footer=180,delimiter=',')
Elements   = np.genfromtxt("B737.inp",skip_header = 6598,skip_footer=1361,delimiter=',')

#averaging the element coordinate
num = 1
Mises_coordinates= []
for i in Elements:
    xyz1         = Displ_coordinates[int(i[1])-1]
    xyz2         = Displ_coordinates[int(i[2])-1]
    xyz3         = Displ_coordinates[int(i[3])-1]
    xyz4         = Displ_coordinates[int(i[4])-1]
    xcoord       =  (xyz1[1]+xyz2[1]+xyz3[1]+xyz4[1])/4
    ycoord       = (xyz1[2]+xyz2[2]+xyz3[2]+xyz4[2])/4
    zcoord       =  (xyz1[3]+xyz2[3]+xyz3[3]+xyz4[3])/4
    node_element = [num , xcoord, ycoord,zcoord]
    Mises_coordinates.append(node_element)
    num         += 1
Mises_coordinates = np.array(Mises_coordinates)


#Get all values
Mises_bending1    = np.genfromtxt("B737.rpt",skip_header=20, skip_footer = 59956-5799-165)
Mises_bending2    = np.genfromtxt("B737.rpt",skip_header=5816, skip_footer = 59956-6673-158)
Mises_Jam1        = np.genfromtxt("B737.rpt",skip_header=6705, skip_footer = 59956-12484-146)
Mises_Jam2        = np.genfromtxt("B737.rpt",skip_header=12501, skip_footer = 59956-13358-139)
Mises_straight1   = np.genfromtxt("B737.rpt",skip_header=13390, skip_footer = 59956-19169-127)
Mises_straight2   = np.genfromtxt("B737.rpt",skip_header=19186, skip_footer = 59956-20043-120)

Displacement_bending1         = np.genfromtxt("B737.rpt",skip_header=20074, skip_footer = 59956-26663-108)
Displacement_bending_nodes    = np.genfromtxt("B737.rpt",skip_header=26678, skip_footer = 59956-26695-101)
Displacement_Jam1             = np.genfromtxt("B737.rpt",skip_header=26724, skip_footer = 59956-33313-89)
Displacement_Jam_nodes        = np.genfromtxt("B737.rpt",skip_header=33328, skip_footer = 59956-33345-82)
Displacement_straight1        = np.genfromtxt("B737.rpt",skip_header=33374, skip_footer = 59956-39963-70)
Displacement_straight_nodes   = np.genfromtxt("B737.rpt",skip_header=39978, skip_footer = 59956-39995-63)
Reaction_bending_nodes        = np.genfromtxt("B737.rpt",skip_header=46628, skip_footer = 59956-46645-44)
Reaction_Jam_nodes            = np.genfromtxt("B737.rpt",skip_header=53278, skip_footer = 59956-53295-25)
Reaction_straight_nodes       = np.genfromtxt("B737.rpt",skip_header=59928, skip_footer = 59956-59945-6)

#Combine region 1 and 2
Mises_bending  = np.vstack((Mises_bending1[0:1732],Mises_bending2[0:409],Mises_bending1[1732:3429],Mises_bending2[409:522],Mises_bending1[3429:4857],Mises_bending2[522:579],Mises_bending1[4857:5656],Mises_bending2[579:860],Mises_bending1[5656:5783]))
Mises_Jam      = np.vstack((Mises_Jam1[0:1732],Mises_Jam2[0:409],Mises_Jam1[1732:3429],Mises_Jam2[409:522],Mises_Jam1[3429:4857],Mises_Jam2[522:579],Mises_Jam1[4857:5656],Mises_Jam2[579:860],Mises_Jam1[5656:5783]))
Mises_straight = np.vstack((Mises_straight1[0:1732],Mises_straight2[0:409],Mises_straight1[1732:3429],Mises_straight2[409:522],Mises_straight1[3429:4857],Mises_straight2[522:579],Mises_straight1[4857:5656],Mises_straight2[579:860],Mises_straight1[5656:5783]))

#Average inside and outside von mises stresses/shear
#node, Von Mises, Shear
Mises_bending[:,2]  = (Mises_bending[:,2]+Mises_bending[:,3])/2
Mises_bending[:,4]  = (Mises_bending[:,4]+Mises_bending[:,5])/2
Mises_bending       = np.delete(Mises_bending,(1,3,5),axis=1)
Mises_Jam[:,2]      = (Mises_Jam[:,2]+Mises_Jam[:,3])/2
Mises_Jam[:,4]      = (Mises_Jam[:,4]+Mises_Jam[:,5])/2
Mises_Jam           = np.delete(Mises_Jam,(1,3,5),axis=1)
Mises_straight[:,2] = (Mises_straight[:,2]+Mises_straight[:,3])/2
Mises_straight[:,4] = (Mises_straight[:,4]+Mises_straight[:,5])/2
Mises_straight      = np.delete(Mises_straight,(1,3,5),axis=1)



#Creating one array of points:Vonmises:shear
Mises_shear_bending = np.hstack((Mises_coordinates,Mises_bending))

print(Mises_coordinates)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = Mises_coordinates[:,1]
ys = Mises_coordinates[:,2]
zs = Mises_coordinates[:,3]
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()