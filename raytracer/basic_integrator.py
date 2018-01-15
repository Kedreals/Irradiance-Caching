#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
"""

from integrator import Integrator
from intersection import Intersection
import numpy as np
from ray import Ray


def getCosineWeightedPointR3(n):
    o = np.zeros(2)
    o[0] = np.arccos(n[2])
    o[1] = np.arctan2(n[1], n[0])

    omega = np.random.rand(2)
    omega[0] = np.arcsin(np.sqrt(omega[0]))
    omega[1] *= np.pi * 2

    RotTheta = np.array([[np.cos(o[0]), 0., np.sin(o[0])], [0., 1., 0.], [-np.sin(o[0]), 0., np.cos(o[0])]])
    RotPhi = np.array([[np.cos(o[1]), np.sin(o[1]), 0.], [-np.sin(o[1]), np.cos(o[1]), 0.], [0., 0., 1.]])

    r = np.zeros(3)
    r[0] = np.sin(omega[0]) * np.cos(omega[1])
    r[1] = np.sin(omega[0]) * np.sin(omega[1])
    r[2] = np.cos(omega[0])

    r = np.dot(RotPhi, np.dot(RotTheta, r))

    return r

def MonteCarlo(intersection, scene, sampleCount=128, sample=None):
    res = 0.0
    minHitDist = np.infty

    for i in range(sampleCount):
        d = getCosineWeightedPointR3(intersection.n)
        r = Ray(intersection.pos + 0.001 * d, d)
        ni = Intersection()
        if (scene.intersect(r, ni)):
            if (r.t < minHitDist):
                minHitDist = r.t
            res += ni.ell * np.pi
            if ((sample != None) & (ni.ell > 0)):
                sample.avgLightDir += d

    res *= 1 / sampleCount
    if (sample != None):
        if (np.linalg.norm(sample.avgLightDir > 0)):
            sample.avgLightDir = sample.avgLightDir / np.linalg.norm(sample.avgLightDir)
        sample.minHitDist = minHitDist
    return np.min([res, 1.0])

class BasicIntegrator(Integrator) :
    def __init__(self):
        self.showSamples = False

    def ell(self, scene, ray, camera):
        intersection = Intersection(np.array([0., 0., 0.]), np.array([0.,1.,0.]))
        intersection.color = np.array([1., 1., 1.])
        if( scene.intersect(ray, intersection)) :
            return np.max([0.0, np.min([1.0, intersection.ell + MonteCarlo(intersection, scene, 256)])])*intersection.color
        
        return np.array([0., 0., 0.])
        
        
        