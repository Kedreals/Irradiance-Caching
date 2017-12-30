#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
"""

from integrator import Integrator
from intersection import Intersection
import numpy as np

class BasicIntegrator(Integrator) :
    
    def ell(self, scene, ray):
        intersection = Intersection(np.array([0., 0., 0.]), np.array([0.,1.,0.]))
        if( scene.intersect(ray, intersection)) :
            return np.max([0.0, np.dot(intersection.n, np.array([0.0, 1., 0.]))])*np.ones(3)
        
        return np.array([0., 0., 0.])
        
        
        