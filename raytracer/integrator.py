#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
"""

from abc import ABCMeta
import numpy as np

class Integrator :
    __metaclass__ = ABCMeta
        
    def ell(self, scene, ray, camera):
        raise NotImplementedError()
        
    def getCosineWeightedPointR3(self, n):
        o = np.zeros(2)
        o[0] = np.arccos(n[2])
        o[1] = np.arctan2(n[1], n[0])

        omega = np.random.rand(2)
        omega[0] = np.arcsin(np.sqrt(omega[0]))
        omega[1] *= np.pi * 2

        RotTheta = np.array([[np.cos(o[0]), 0., np.sin(o[0])], [0., 1., 0.], [-np.sin(o[0]), 0., np.cos(o[0])]])
        RotPhi = np.array([[np.cos(o[1]), -np.sin(o[1]), 0.], [np.sin(o[1]), np.cos(o[1]), 0.], [0., 0., 1.]])

        r = np.zeros(3)
        r[0] = np.sin(omega[0]) * np.cos(omega[1])
        r[1] = np.sin(omega[0]) * np.sin(omega[1])
        r[2] = np.cos(omega[0])

        r = np.dot(RotPhi, np.dot(RotTheta, r))

        if np.dot(n, r) < 0:
            print("d=", r, ", n=", n)

        return r