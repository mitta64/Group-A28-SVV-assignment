import matplotlib.pyplot as plt
import numpy as np

def integrate_z(grid):
  """used to integrate the .dat aero data over the x-axis"""
  Ca = 0.505
  h_res = 41
  v_res = 81
  
  solution = []
  for column in range(len(grid[0])):
    A = 0
    for row in range(len(grid)):
      A += grid[row][column]/v_res*Ca

    solution.append(A)
  return solution

def integrate_x(x_list):
  """used to integrate the .dat aero data over the x-axis"""
  Ca = 0.505
  span = 1.611
  h_res = 41
  v_res = 81
  
  solution = []
  prev = 0
  value = 0
  for element in range(len(x_list)):
    value += (prev+x_list[element])/2*(span/h_res)
    prev=x_list[element]
    
    solution.append(value)
  return solution

def def_integral(f,x1,x2,res=10000):
    interval = (x2-x1)/res
    solution = 0
    a=f(x1)
    for e in range(res):
        b=f(x1+(e+1)*interval)
        solution += (a+b)*interval/2
        a=b
    return solution

def indef_integral(f,x1,x2,res=10000):
    interval = (x2-x1)/res
    solution = []
    value = 0
    a = f(x1)
    for e in range(res):
        b = f(x1+(e+1)*interval)
        value += (a+b)*interval/2
        solution.append(value)
        a = b
    return solution


""" How to use: """
##grid = aero_data()
##int_1 = integrate_z(grid)
##int_2 = integrate_x(int_1)
##int_3 = integrate_x(int_2)
##int_4 = integrate_x(int_3)
##int_5 = integrate_x(int_4)

##def function(x):
##  return x
##x = np.linspace(0,1.611,1000)
##y = indef_integral(function,0,1.611,1000)
##
##plt.xlim([0,1.611])
##plt.xlabel('x-axis')
##plt.ylabel('z-axis')
##plt.plot()
##plt.show()
