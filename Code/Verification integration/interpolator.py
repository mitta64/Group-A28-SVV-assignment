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



