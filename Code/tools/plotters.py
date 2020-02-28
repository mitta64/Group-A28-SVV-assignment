# Plotting functions
import matplotlib.pyplot as plt
import numpy as np



def plot(data, thing_to_plot, unit):
  """ Plot deflection or twist data on a 2D graph
        thing_to_plot and unit should be written as strings, like 'deflection', '"""
  x=np.linspace(0,1.611,len(data))
  plt.plot(x,data)
  plt.xlabel('span (m)')
  plt.ylabel(thing_to_plot + ' (' + unit + ')')
  plt.show()
  return()

##func = np.sin(np.linspace(0,10,100))
##thing = 'deflection'
##unit = 'm'
##plot(func, thing, unit)


def cross_section_plot(data, thing_to_plot, unit):
  """ Plots a cross section of the aileron which represents stresses in terms of colors """
  
  points = []
  point1 = [-C_a,0]
  point2 = [-h,-h/2]
  point3 = [-h,h/2]

  for i in range(1,3):
    plt.plot([eval('point' + str(i))[0],eval('point' + str(i+1))[0]],[eval('point' + str(i))[1],eval('point' + str(i+1))[1]])
  plt.plot([eval('point' + str(1))[0],eval('point' + str(3))[0]],[eval('point' + str(1))[1],eval('point' + str(3))[1]])
  plt.xlim(0,-0.505)
  plt.show()
  return()


thing = 'deflection'
unit = 'm'
cross_section_plot(12, thing, unit)





