import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def nodenumberfun(node,node_i):
    for i in node:
        nodenumber = 0
        if node_i<= i:  #inter_value>node[-2]: #check at which spline to interpolate
            break
        if node_i >= node[-2]:
            nodenumber = len(node)-1
            break
        else:
            nodenumber+=1
    return nodenumber

def cubic_coefficients(node,value):
    # IMPORTANT: needs a grid in chronological order (from small to big)
    #This function creates a matrix containing all the splines coefficients for every node,
    #This way the main calculation only has to be done once, and spline_interpolator actually computes the value
    #input: nodes (1d list), value at these nodes (1dlist), boundary 1 (f'(0)=?),boundary 2 (f'(n)=?)
    #output Array containing Splinematrix
    #boundary1 = (value[1]-value[0])/(node[1]-node[0])
    #boundary2 = (value[-1]-value[-2])/(node[-1]-node[-2])
    # print(boundary1,boundary2)
    Mmatrix = []
    dmatrix = []
    Lambda0 = 1
    #boundary 1
    Mmatrix.append(list(np.concatenate((np.array([2,0]),np.zeros(len(node)-2)),axis=0)))
    dmatrix.append(0)#(((value[1]-value[0])-boundary1)/(node[1]-node[0]))/(node[1]-node[0]))
    for i in range(1,len(node)-1):
        #Main matrix
        hi      =  node[i]-node[i-1]
        hi1     =  node[i+1]-node[i]
        mui     = hi/(hi+hi1)
        Lambdai =  1-mui
        a       = np.zeros(i-1)
        if len(node)-len(a)-3>=0:
            b   = np.zeros(len(node)-len(a)-3)
        else:
            b   = np.array([])
        Mmatrix.append(list(np.concatenate((a,np.array([mui,2,Lambdai]),b),axis=0)))
        #outcome matrix
        f = ((value[i+1]-value[i])/(node[i+1]-node[i])-(value[i]-value[i-1])/(node[i]-node[i-1]))/(node[i+1]-node[i-1])
        dmatrix.append(f)
    mun=1
    #boundary
    Mmatrix.append(list(np.concatenate((np.zeros(len(node) - 2),np.array([0,2])), axis=0)))
    dmatrix.append(0)#(boundary2-(value[-1] - value[-2]) / (node[-1] - node[-2])) / (node[-1] - node[-2]))
    #solve for coefficients
    dmatrix = 6*np.array(dmatrix)
    coefficients = np.linalg.solve(Mmatrix,dmatrix)
    return coefficients


def cubic_interpolator(coefficients, node, value, inter_node):
    # This function actually interpolates (1 point)
    # input Splinematrix from previous function, all nodes (1d array), intervalue (the point to be interpolated)
    nodenumber=0
    for i in node:
        if inter_node<= i:  #inter_value>node[-2]: #check at which spline to interpolate
            break
        if inter_node >= node[-2]:
            nodenumber = len(node)-1
            break
        else:
            nodenumber+=1
    nodenumber = nodenumber #no
    xi = node[nodenumber]
    x_i = node[nodenumber-1]
    yi = value[nodenumber]
    y_i = value[nodenumber-1]
    hi = xi - x_i

    a = coefficients[nodenumber-1]/(6*hi)
    b =  coefficients[nodenumber]/(6*hi)
    c = coefficients[nodenumber-1]*hi*hi/6
    d = coefficients[nodenumber]*hi*hi/6
    si = a*(xi-inter_node)**3+b*(inter_node-x_i)**3+(y_i-c)*(xi-inter_node)/hi+(yi-d)*(inter_node-x_i)/hi
    return si

def interpolate_val(Data, numxentries, nodex, valueentry, inter_node):
    #COMPUTES DATA VALUE AT GIVEN X SLICE
    #INPUT DATA (THE TO BE INTERPOLATED ARRAY (E.G. MISES_SHEAR_BENDING),NUMXENTRIES (NUMBER OF NODES PER SLICE), NODEX(WHICH SLICE)
    #VALUEENTRY(WHICH COLUMN CONTAINS THE TO BE INTERPOLATED DATA), INTERNODE (Y AND Z COORDINATES)
    #inter_node = np.sqrt(inter_node[0]**2+inter_node[1]**2)
    test = Data[0+nodex*numxentries:(nodex+1)*numxentries]
    spar = []
    upper = []
    lower = []
    for i in test:
        if float(i[3])==0.0:
            spar.append(list(i))
        if i[2] <= 0 and i[3] != 0:
            lower.append(list(i))
        if i[2] > 0 and i[3] != 0:
            upper.append(list(i))
    spar = np.array(spar)
    upper = np.array(upper)
    lower = np.array(lower)
    spar  = spar[spar[:,2].argsort()]
    upper = upper[upper[:,3].argsort()]
    lower = lower[lower[:,3].argsort()]
    #print(lower)
    #chord = np.vstack((np.array(lower),np.array(upper)))
    chordcoord = [0]
    dis = 0

    if inter_node[0]==0:
        inter_nodeint = inter_node[0]
        cof = cubic_coefficients(spar[2], spar[:, valueentry])
        Fr = cubic_interpolator(cof, spar[2], spar[:, valueentry], inter_nodeint)
    else:
        if inter_node[0]<0 :
            for j in range(len(lower) - 1):
                dis = np.sqrt((lower[j + 1, 2] - lower[j, 2]) ** 2 + (lower[j + 1, 3] - lower[j, 3]) ** 2) + dis
                chordcoord.append(dis)
            nodenumber = nodenumberfun(lower[:,3], inter_node[1])
            inter_nodeint = chordcoord[nodenumber] + np.sqrt((inter_node[0]-lower[nodenumber,2])**2+(inter_node[1]-lower[nodenumber,3])**2)
            cof = cubic_coefficients(chordcoord, lower[:, valueentry])
            Fr = cubic_interpolator(cof, chordcoord, lower[:, valueentry], inter_nodeint)
        if inter_node[0]>0 :
            for j in range(len(upper) - 1):
                dis = np.sqrt((upper[j + 1, 2] - upper[j, 2]) ** 2 + (upper[j + 1, 3] - upper[j, 3]) ** 2) + dis
                chordcoord.append(dis)
            nodenumber = nodenumberfun(lower[:, 3], inter_node[1])
            inter_nodeint = chordcoord[nodenumber] + np.sqrt((inter_node[0] - lower[nodenumber, 2]) ** 2 + (inter_node[1] - lower[nodenumber, 3]) ** 2)
            cof = cubic_coefficients(chordcoord, upper[:, valueentry])
            Fr = cubic_interpolator(cof, chordcoord, upper[:, valueentry], inter_nodeint)

    return Fr


#-----------------------------------------------------------------------------------------------------------------------------------


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



#-----------------------------------------------------------------------------------------------------------------------

#These are the final to be used arrays
#ENTRY = [#node, x , y ,z , values]
#Creating one array of points:Vonmises:shear
Mises_shear_bending         = np.hstack((Mises_coordinates,Mises_bending[:,1:]))
Mises_shear_Jam             = np.hstack((Mises_coordinates,Mises_Jam[:,1:]))
Mises_shear_straight        = np.hstack((Mises_coordinates,Mises_straight[:,1:]))
Displacement_bending        = np.hstack((Displ_coordinates,Displacement_bending1[:,1:]))
Displacement_Jam            = np.hstack((Displ_coordinates,Displacement_Jam1[:,1:]))
Displacement_straight       = np.hstack((Displ_coordinates,Displacement_straight1[:,1:]))
Displacement_bending_nodes  = np.hstack((Boundarynode_coordinates,Displacement_bending_nodes[:,1:]))
Displacement_Jam_nodes      = np.hstack((Boundarynode_coordinates,Displacement_Jam_nodes[:,1:]))
Displacement_straight_nodes = np.hstack((Boundarynode_coordinates,Displacement_straight_nodes[:,1:]))
Reaction_bending_nodes      = np.hstack((Boundarynode_coordinates,Reaction_bending_nodes[:,1:]))
Reaction_Jam_nodes          = np.hstack((Boundarynode_coordinates,Reaction_Jam_nodes[:,1:]))
Reaction_straight_nodes     = np.hstack((Boundarynode_coordinates,Reaction_straight_nodes[:,1:]))



#sorting in x: 62 nodes per slice Mises, 61 nodes per slice Displacement, 108 slices
Mises_shear_bending          = Mises_shear_bending[Mises_shear_bending[:,1].argsort()]
Mises_shear_Jam              = Mises_shear_Jam[Mises_shear_Jam[:,1].argsort()]
Mises_shear_straight         = Mises_shear_straight[Mises_shear_straight[:,1].argsort()]
Displacement_bending         = Displacement_bending[Displacement_bending[:,1].argsort()]
Displacement_Jam             = Displacement_Jam[Displacement_Jam[:,1].argsort()]
Displacement_straight        = Displacement_straight[Displacement_straight[:,1].argsort()]
Displacement_bending_nodes   = Displacement_bending_nodes[Displacement_bending_nodes[:,1].argsort()]
Displacement_Jam_nodes       = Displacement_Jam_nodes[Displacement_Jam_nodes[:,1].argsort()]
Displacement_straight_nodes  = Displacement_straight_nodes[Displacement_straight_nodes[:,1].argsort()]
Reaction_bending_nodes       = Reaction_bending_nodes[Reaction_bending_nodes[:,1].argsort()]
Reaction_Jam_nodes           = Reaction_Jam_nodes[Reaction_Jam_nodes[:,1].argsort()]
Reaction_straight_nodes      = Reaction_straight_nodes[Reaction_straight_nodes[:,1].argsort()]

#Meshes
gridMises_shear_bending          = Mises_shear_bending[:,1:4]
gridMises_shear_Jam              = Mises_shear_Jam[:,1]
gridMises_shear_straight         = Mises_shear_straight[:,1:4]
gridDisplacement_bending         = Displacement_bending[:,1:4]
gridDisplacement_Jam             = Displacement_Jam[:,1:4]
gridDisplacement_straight        = Displacement_straight[:,1:4]
gridDisplacement_bending_nodes   = Displacement_bending_nodes[:,1:4]
gridDisplacement_Jam_nodes       = Displacement_Jam_nodes[:,1:4]
gridDisplacement_straight_nodes  = Displacement_straight_nodes[:,1:4]
gridReaction_bending_nodes       = Reaction_bending_nodes[:,1:4]
gridReaction_Jam_nodes           = Reaction_Jam_nodes[:,1:4]
gridReaction_straight_nodes      = Reaction_straight_nodes[:,1:4]

y= []
x = []

for i in range(107):
    a = interpolate_val(Displacement_Jam, 61, i, 6, [0,10.25])
    x.append(Displacement_Jam[i*62,1])
    y.append(a)

fig= plt.figure()
plt.plot(x,y)
plt.xlabel("x(mm)")
plt.ylabel("Displacement(mm)")




#figures for halfway the aileron
#Bending



plt.xlabel("x (mm)")
plt.ylabel("Displacement (mm)")

ys = np.array(Mises_shear_bending[53*62:54*62,2])
zs = np.array(Mises_shear_bending[53*62:54*62,3])

fig, axs = plt.subplots(2, 2)

axs[0,0].scatter(ys,zs, c = np.array(Mises_shear_bending[53*62:54*62,4]),cmap = "seismic")
#axs[0,0].colorbar()


axs[1,0].scatter(ys,zs, c = np.array(Mises_shear_bending[53*62:54*62,5]),cmap = "seismic")
#axs[1,0].colorbar()

ys = np.array(Displacement_bending[53*61:54*61,2])
zs = np.array(Displacement_bending[53*61:54*61,3])


axs[0,1].scatter(ys,zs, c = np.array(Displacement_bending[53*61:54*61,4]),cmap = "seismic")
#axs[0,1].colorbar()




plt.show()


fig = plt.figure()

#ax = fig.add_subplot(111, projection='3d')
xs = np.array(Mises_shear_bending[:,1])
ys = np.array(Mises_shear_bending[:,2])
zs = np.array(Mises_shear_bending[:,3])
plt.scatter(ys,zs, c = np.array(Mises_shear_bending[:,5]),cmap = "seismic")

#ax.scatter(xs,ys, zs, c = np.array(Mises_shear_bending[:,5]),cmap = "seismic")
#plt.scatter(ys, zs, c = np.array(Mises_shear_bending[50*62:51*62,4]),cmap = "seismic")
#plt.colorbar()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = np.array(Displacement_Jam[:,1])
ys = np.array(Displacement_Jam[:,2])
zs = np.array(Displacement_Jam[:,3])

ax.scatter(xs,ys, zs, c = np.array(Displacement_Jam[:,4]),cmap = "seismic")

plt.show()




#----------------------------------------------------------------------------------------
#Error calculations:
NumresMises_shear = []

#err = (NumresMises_shear-Mises_shear_bending)
#err = np.linalg.norm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = np.array(Displacement_Jam[:,1])
ys = np.array(Displacement_Jam[:,2])
zs = np.array(Displacement_Jam[:,3])

ax.scatter(xs,ys, zs, c = np.array(Err),cmap = "seismic")

