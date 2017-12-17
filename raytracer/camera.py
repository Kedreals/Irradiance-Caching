#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:26:46 2017

@author: lessig
"""

import numpy as np
import ray

class Camera :

    def __init__( self, res_x = 512, res_y = 512):
        self.pos = np.zeros( 3)
        self.viewdir = np.array( [0.0, 1.0, 0.0])
        self.updir = np.array( [0.0, 0.0, 1.0])
        self.fov = (30.0 * np.pi) / 180.0
        self.flength = 1.0
        self.image = np.zeros( [res_x, res_y])
    
    def generateRay( self, pix_x, pix_y) :
        
        # compute direction
        xmax = np.sin( self.fov) * self.flength
        dx = -xmax + pix_x * (2 * xmax) / (self.image.shape[0])
        dy = -xmax + pix_y * (2 * xmax) / (self.image.shape[1])
        
        dir = np.array([dx, dy, 1.0])
        dir = dir / np.linalg.norm( dir)
        
        r = ray.Ray( self.pos, dir)
        return r
    