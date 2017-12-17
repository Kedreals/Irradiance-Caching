#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
"""

from abc import ABCMeta

class Shape :
    __metaclass__ = ABCMeta
    
    objectid = 0
    
    def __init__(self) :
        Shape.objectid += 1
    
    def intersect(self, ray):
        raise NotImplementedError()
        
        