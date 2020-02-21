# Aero data image

from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv


# Read the files
file="aerodynamicloadf100.dat"
csv=open(file, "r").read()


table = csv.split("\n")
table=table[:-1]

number_table=[]
for element in table:
  number_table.append(eval(element))

row=[]
image=[]
for r in number_table:
  for e in r:
    row.append(float(e*10**3))
  image.append(row)
  row=[]

##def matrix(n,m=0,e=0):
##    if m==0:
##        m=n
##    A=[]
##    for i in range(n):
##        A.append([e]*m)
##    return A
##
##def transpose(A):
##    if type(A[0])==int or type(A[0])==float:
##        return vector(A)
##
##    T=matrix(len(A[0]),len(A))
##    for i in range(len(A)):
##        for j in range(len(A[0])):
##            T[j][i]=A[i][j]
##    return T
##image=transpose(image)

##width = len(image[0])
##height = len(image)


coord_sys=(1.611,0,-0.505,0)
plt.xlabel('x-axis')
plt.ylabel('z-axis')

plt.imshow(image, extent=coord_sys,interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()


##def average(row):
##  av=0
##  for e in row:
##    av+=e
##  av/=len(row)
##  return av
##y=[]
##for row in image:
##  y.append(average(row))
##x=arange(0,81)
##
##plt.plot(x,y)
##plt.show()
