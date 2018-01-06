#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
"""

from abc import ABCMeta
import numpy as np

class Shape :
    __metaclass__ = ABCMeta
    
    objectid = 0
    
    def __init__(self, ell=0.0, color=np.array([1., 1., 1.])):
        self.ell = ell
        self.color = color
        Shape.objectid += 1
    
    def intersect(self, ray, intersection):
        raise NotImplementedError()
        
        