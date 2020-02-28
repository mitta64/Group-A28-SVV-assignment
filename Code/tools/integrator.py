import matplotlib.pyplot as plt
import numpy as np



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
