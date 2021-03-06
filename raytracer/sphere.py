#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:58:48 2017

@author: lessig
"""

import numpy as np
from shape import Shape


class Sphere( Shape) :
    
    def __init__(self, pos, r, ell=0, color=np.array([1., 1., 1.])):
    
        super().__init__(ell, color)
        
        self.pos = pos
        self.r = r
        
    def intersect(self, ray, intersection):
        
        # compute intersection point with sphere
        q = ray.o - self.pos
        
        c = np.dot( q, q) - (self.r * self.r)
        b = 2.0 * np.dot( q, ray.d)
        
        temp = b*b - 4*c 
        if( temp < 0.0) :
            return False
        
        temp = np.sqrt( temp)
        s1 = 1.0/2.0 * ( -b + temp) 
        s2 = 1.0/2.0 * ( -b - temp)
        
        sol = s1
        if s1 <= 0.0 and s2 <= 0.0:
            return False
        if s1 <= 0.0:
            sol = s2
        elif s2 <= 0.0:
            sol = s1

        elif s2 < s1:
            sol = s2
            
        if sol < ray.t :
            ray.t = sol;
            intersection.pos = ray.o + ray.t * ray.d
            intersection.n = +intersection.pos - self.pos
            intersection.n /= np.linalg.norm(intersection.n)
            intersection.ell = self.ell
            intersection.color = self.color
            return True

        return False
        
        
    