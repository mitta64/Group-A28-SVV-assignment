import numpy as np 
import matplotlib.pyplot as plt 

def retrieve_aero_data(): 
  file="aerodynamicloadf100.dat" 
  csv=open(file, "r").read() 
  table = csv.split("\n") 
  table=table[:-1] 
  number_table=[] 
  for element in table: 
    number_table.append(eval(element)) 
  row=[] 
  grid=[] 
  for r in number_table: 
    for e in r: 
      row.append(float(e)*10**3) 
    grid.append(row) 
    row=[] 
  return grid 
 
#matrix with n rows, m columns 
def matrix(n,m=0,e=0): 
    if m==0: 
        m=n 
    A=[] 
    for i in range(n): 
        A.append([e]*m) 
    return A 
 
# turns list into vector 
def vector(lst): 
    n=len(lst) 
    B=matrix(n,1) 
    for i in range(n): 
        B[i][0]=lst[i] 
    return B 
 
# transpose of a matrix 
def transpose(A): 
    if type(A[0])==int or type(A[0])==float: 
        return vector(A) 
 
    T=matrix(len(A[0]),len(A)) 
    for i in range(len(A)): 
        for j in range(len(A[0])): 
            T[j][i]=A[i][j] 
    return T 
 
 
def times_z(aero_data, nodes_z, z_sc): 
    aero_data_z = [] 
    for row in range(len(aero_data)): 
        factor = nodes_z[row] + z_sc 
#        print(factor)
        new_row = np.array(aero_data[row])*factor
#        print(new_row)
        aero_data_z.append(new_row) 
#        if factor > 0:
#            exit()
#    print(aero_data_z[0])
    return aero_data_z 
 
 
pi=3.14159265359 
Nz, Nx = 81, 41 
Ca, la = 0.505, 1.611 
 
theta_z, nodes_z = [], [] 
for i in range(1,Nz+2): 
    theta_z.append((i-1)/Nz*pi) 
for t in range(len(theta_z)-1): 
    nodes_z.append(1/2*(Ca/2*(1-np.cos(theta_z[t]))+Ca/2*(1-np.cos(theta_z[t+1])))) 
 
theta_x, nodes_x = [], [] 
for i in range(1,Nx+2): 
    theta_x.append((i-1)/Nx*pi) 
for t in range(len(theta_x)-1): 
    nodes_x.append(1/2*(la/2*(1-np.cos(theta_x[t]))+la/2*(1-np.cos(theta_x[t+1])))) 

 

aero_data = retrieve_aero_data() 
grid = transpose(aero_data) 
 
