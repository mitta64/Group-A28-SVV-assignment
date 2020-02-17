# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:16:23 2020

@author: Group A28 
"""
#=======================================================================================
"Cross sectional properties for bending"
"Requirement: Make it suitable for a box and an aileron cross section"

class CrossSectionalProperties(object):
    """A class that computes:
        
        - Boom Areas & Locations
        - Centroid
        - Second Moment of Inertias I_xx, I_yy, I_xy
        - Shear Centre
        - Torsional Stiffness
        
        #==========================    
        Outputs: Boom Areas & Boom Locations
        Inputs: - Chord length aileron
                - Height aileron
                - Spar thickness
                - Stringer spacing delta_st
                - Number of stringers n_st
                - Stringer locations
                - Skin thickness
                - Stringer height
                - Stringer width
                - Stringer thickness 
        #==========================
        Output: Centroid
        Inputs: See input list of Boom Areas
        #==========================    
        Outputs: Second Moment of Inertias I_xx, I_yy, I_xy
        Inputs: See input list of Boom Areas
        #==========================    
        Output: Shear Centre
        Inputs: - Boom areas 
                - Boom locations 
                - Skin thickness
        #==========================    
        Output: Torsional Stiffness
        Inputs: - Shear flow distributions
        #==========================  
        Output: Visual representation of cross section
                    -> Black colour lines for the skin
                    -> Red dots for booms
        Inputs: - Aileron height
                - Aileron chord length
                - Skin thickness
                - Boom areas
                - Boom locations
        #==========================  
    """
    #========================       
    #Compute Boom Areas & Boom Locations
    #========================
    def booms(self):
        
        
        
        
    
    #========================       
    #Compute Centroid
    #========================
    def centroid(self):
        
        
    #========================       
    #Compute Second Moment of Inertia
    #========================
    def second_moi(self):
    
    
    
    
    #I_xx
    
    
    #I_yy
    
    
    #I_xy
    
    #========================       
    #Compute Shear Centre
    #========================
    # Requirements:
        # Locations of the booms
        # Skin thickness
        # Skin Locations
    def shear_centre(self):
        
        
    
    #========================       
    #Compute Torsional Stiffness
    #========================
    def torsional_stiffness(self):
        
    
    
    
    
    

    



#=======================================================================================
"Integrator"

def integral(f,x1,x2,res=10000):
    i=(x2-x1)/res   # interval
    A=0
    a=f(x1)
    for e in range(res):
        b=f(x1+(e+1)*i)
        A+=(a+b)*i/2
        a=b
    return A