# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:16:34 2020

@author: chang
"""
import re

import numpy as np

xyzcoord = open("gridxyz.txt", "r")
correct = []
lines = xyzcoord.readlines()
x,y,z = [], [], []
for line in lines:
    correct += filter(None,re.split(r'\W|\d', line))
    
#    for i in thisline:
#        
#        x.append(thisline[0])
#        y.append(thisline[1])
#        z.append(thisline[2])