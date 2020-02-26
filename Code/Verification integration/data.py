# Retrieve all data 
import numpy as np 
import matplotlib.pyplot as plt 
 
name = 'f100' 
C_a = 0.505  # m 
l_a = 1.611  # m 
x_1 = 0.125  # m 
x_2 = 0.498  # m 
x_3 = 1.494  # m 
x_a = 0.245   # m 
h = 0.161  # m 
t_sk = 1.1/1000  # m 
t_sp = 2.4/1000  # m 
t_st = 1.2/1000  # m 
h_st = 13/1000   # m 
w_st = 17/1000   # m 
n_st = 11  # - 
d_1 = 0.00389  # m 
d_3 = 0.01245  # m 
theta = 0 
P = 49.2*1000  # N 
 
class Aircraft(object): 
    def __init__(self,name,C_a,l_a,x_1,x_2,x_3,x_a,h,t_sk,t_sp,t_st,h_st,w_st,n_st,d_1,d_3,theta,P): 
        self.name = name 
        self.C_a = C_a          #"Chord length aileron[m]" 
        self.l_a = l_a          #"Span of the aileron[m]" 
        self.x_1 = x_1          #"x-location of hinge 1 [m]" 
        self.x_2 = x_2          #"x-location of hinge 2 [m]" 
        self.x_3 = x_3          #"x-location of hinge 3 [m]" 
        self.x_a = x_a          #"Distance between actuator 1 and 2 [cm]" 
        self.h = h              #"Aileron height[cm]" 
        self.t_sk = t_sk        #"Skin thickness [mm]" 
        self.t_sp = t_sp        #"Spar thickness [mm]" 
        self.t_st = t_st        #"Thickness of stiffener[mm]" 
        self.h_st = h_st        #"Height of stiffener[cm]" 
        self.w_st = w_st        #"Width of stiffener[cm]" 
        self.n_st = n_st        #"Number of stiffeners [-]" 
        self.d_1 = d_1          #"Vertical displacement hinge 1[cm]" 
        self.d_3 = d_3          #"Vertical displacement hinge 3[cm]" 
        self.theta = theta      #"Maximum upward deflection[deg]" 
        self.P = P              #"Load in actuator 2[kN]" 
 
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
        new_row = np.array(aero_data[row])*factor 
        aero_data_z.append(new_row) 
        grid_z = transpose(aero_data_z) 
    return grid_z 
 
 
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
 
 
f100 = Aircraft(name,C_a,l_a,x_1,x_2,x_3,x_a,h,t_sk,t_sp,t_st,h_st,w_st,n_st,d_1,d_3,theta,P) 
aero_data = retrieve_aero_data() 
grid = transpose(aero_data) 
 
 
 
