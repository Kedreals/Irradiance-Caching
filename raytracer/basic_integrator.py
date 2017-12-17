#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
"""

from integrator import Integrator


class BasicIntegrator(Integrator) :
    
    def ell(self, scene, ray):
        
        if( scene.intersect(ray)) :
            return 1.0
        
        return 0.0
        
        
        