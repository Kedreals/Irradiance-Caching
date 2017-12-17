#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:07 2017

@author: lessig
"""

class Scene :
    
    def __init__(self) :
        self.objects = []
        
    def intersect( self, ray) :
        
        res = False
        for obj in self.objects :
            res |= obj.intersect( ray)
            
        return res
            
        