import numpy as np

#Get nodes
node_coordinates = np.genfromtxt("B737.inp",comments = "*",skip_footer=7996,delimiter=',')
node_coordinates = node_coordinates[:,1:4]
print(node_coordinates[29])

#Get all values
Mises_bending1    = np.genfromtxt("B737.rpt",comments = "*",skip_header=20, skip_footer = 59956-5799)
Mises_bending2    = np.genfromtxt("B737.rpt",comments = "*",skip_header=5816, skip_footer = 59956-6673)
Mises_Jam1        = np.genfromtxt("B737.rpt",comments = "*",skip_header=6705, skip_footer = 59956-12484)
Mises_Jam2        = np.genfromtxt("B737.rpt",comments = "*",skip_header=12501, skip_footer = 59956-13358)
Mises_straight1   = np.genfromtxt("B737.rpt",comments = "*",skip_header=13390, skip_footer = 59956-19169)
Mises_straight2   = np.genfromtxt("B737.rpt",comments = "*",skip_header=19186, skip_footer = 59956-20043)

Displacement_bending1         = np.genfromtxt("B737.rpt",comments = "*",skip_header=20074, skip_footer = 59956-26663)
Displacement_bending_nodes    = np.genfromtxt("B737.rpt",comments = "*",skip_header=26678, skip_footer = 59956-26695)
Displacement_Jam1             = np.genfromtxt("B737.rpt",comments = "*",skip_header=26724, skip_footer = 59956-33313)
Displacement_Jam_nodes        = np.genfromtxt("B737.rpt",comments = "*",skip_header=33328, skip_footer = 59956-33345)
Displacement_straight1        = np.genfromtxt("B737.rpt",comments = "*",skip_header=33374, skip_footer = 59956-39963)
Displacement_straight2        = np.genfromtxt("B737.rpt",comments = "*",skip_header=39978, skip_footer = 59956-39995)

Reaction_bending_nodes        = np.genfromtxt("B737.rpt",comments = "*",skip_header=46628, skip_footer = 59956-46645)
Reaction_Jam_nodes            = np.genfromtxt("B737.rpt",comments = "*",skip_header=53278, skip_footer = 59956-53295)
Reaction_straight_nodes       = np.genfromtxt("B737.rpt",comments = "*",skip_header=59928, skip_footer = 59956-59945)

#Combine region 1 and 2
Mises_bending = np.vstack((Mises_bending1[0:1732],Mises_bending2[0:409],Mises_bending1[1732:3429],Mises_bending2[409:522],Mises_bending1[3429:4857],Mises_bending2[522:579],Mises_bending1[4857:5656],Mises_bending2[579:860],Mises_bending1[5656:5783]))
print(len(node_coordinates))