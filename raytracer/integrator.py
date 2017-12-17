#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
"""

from abc import ABCMeta

class Integrator :
    __metaclass__ = ABCMeta
        
    def ell(self, scene, ray):
        raise NotImplementedError()
        
        