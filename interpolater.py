# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:37:42 2020

@author: changkyupark
"""
import numpy as np

def lagrange_basis(nodes, x=None):
    if x is None: 
        x = nodes
    if isinstance(nodes, list):
        nodes = np.array(nodes)
    p = np.size(nodes)
    basis = np.ones((p, np.size(x)))
    # lagrange basis functions
    for i in range(p):
        for j in range(p):
            if i != j:
                basis[i, :] *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return basis