import numpy as np

"Interpolator"
def spline_coefficient(node,value):
    # IMPORTANT: needs a grid in chronological order (from small to big)
    #This function creates a matrix containing all the splines coefficients for every node,
    #This way the main calculation only has to be done once, and spline_interpolator actually computes the value
    #input: nodes (1d list), value at these nodes (1dlist)
    #output Array containing Splinematrix

    Splinematrix = []
    for i in range(len(node)-1):
        "si = a +b(x-c)"
        a = value[i]
        b=(value[i+1]-value[i])/(node[i+1]-node[i])
        c = node[i]
        #print(a,b,c)
        Splinematrix.append([a,b,c])
    return np.array(Splinematrix)

def spline_interpolator(Splinematrixx, node, inter_node):
    # This function actually interpolates (1 point)
    # input Splinematrix from previous function, all nodes (1d array), intervalue (the point to be interpolated)
    # output value function at node

    nodenumber=0
    for i in node:
        if inter_node<= node[nodenumber] or inter_node>node[-2]: #check at which spline to interpolate
            break
        else:
            nodenumber+=1
    nodenumber = nodenumber-1 #no
    a= Splinematrixx[nodenumber,0]
    b=Splinematrixx[nodenumber,1]
    c=Splinematrixx[nodenumber,2]
    si = a + b*(inter_node-c)
    return si


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
