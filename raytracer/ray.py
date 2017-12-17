#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:26:46 2017

@author: lessig
"""

import numpy as np

class Ray :

    def __init__(self, oa = np.array([0,0,0]), da = np.array([1,0,0])) :
        self.o = oa
        self.d = da
        self.t = np.float64(10000.0)
        
        
