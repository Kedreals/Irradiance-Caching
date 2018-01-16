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
from multiprocessing import Pool


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


def GenRay(intersect):
    d = getCosineWeightedPointR3(intersect.n)
    return (Ray(intersect.pos + 0.001 * d, d), Intersection())


def SceneIntersectList(scene, RayIntersectionTupel):
    return scene.intersect(RayIntersectionTupel[0], RayIntersectionTupel[1])


def GetL(RayIntersectionTupel):
    return RayIntersectionTupel[1][1].ell


def Lamb(x):
    return SceneIntersectList(x[0], x[1])


def MonteCarlo(intersection, scene, sampleCount=128):
    res = 0.0

    Rays = [(scene, GenRay(intersection)) for i in range(sampleCount)]
    b = list(map(Lamb, Rays))
    res = sum(list(map(GetL, Rays))) * np.pi

    """
    for i in range(sampleCount):
        d = getCosineWeightedPointR3(intersection.n)
        r = Ray(intersection.pos + 0.001 * d, d)
        ni = Intersection()
        if (scene.intersect(r, ni)):
            res += ni.ell * np.pi
    """

    res *= 1 / sampleCount

    return res

def getIntersection(sceneRay):
    i = Intersection()
    if(sceneRay[0].intersect(sceneRay[1], i)):
        return i

    return None

def genMonteCarloRays(SceneIntersectionSamplecount):
    """
    for i in range(SceneIntersectionSamplecount[2]):
        SceneIntersectionSamplecount[3].append((SceneIntersectionSamplecount[0], GenRay(SceneIntersectionSamplecount[1]), SceneIntersectionSamplecount[1], SceneIntersectionSamplecount[2]))
    """
    return [(SceneIntersectionSamplecount[0], GenRay(SceneIntersectionSamplecount[1]), SceneIntersectionSamplecount[1], SceneIntersectionSamplecount[2]) for i in range(SceneIntersectionSamplecount[2])]


def TraceMontecCarloRay(SceneRayIntersectionSamplecount):
    I = Intersection()
    res = 0.0
    if(SceneRayIntersectionSamplecount[0].intersect(SceneRayIntersectionSamplecount[1], I)):
        res += I.ell * np.pi

    return (res/SceneRayIntersectionSamplecount[3], SceneRayIntersectionSamplecount[2])

class BasicIntegrator(Integrator):
    def __init__(self):
        self.showSamples = False
        #self.pool = Pool(8)

    def ell(self, scene, ray, camera):
        intersection = Intersection(np.array([0., 0., 0.]), np.array([0., 1., 0.]))
        intersection.color = np.array([1., 1., 1.])
        if (scene.intersect(ray, intersection)):
            return np.max([0.0, np.min(
                [1.0, intersection.ell + MonteCarlo(intersection, scene, 256)])]) * intersection.color

        return np.array([0., 0., 0.])


    def ell2(self, scene, rays, sampleCount=256):
        R = [(scene, rays[i]) for i in range(len(rays))]
        I = self.pool.map(getIntersection, R)
        MRs = []
        IC = [(scene, i, sampleCount, MRs) for i in I if i is not None]


        print("Intersections Calculated")

        MRs = self.pool.map(genMonteCarloRays, IC)

        print("Monte Carlo rays calculated")

        FMRs = sum(MRs, [])

        print("Array flatted")


        R = self.pool.map(TraceMontecCarloRay, FMRs)

        print("finished tracing Monte Carlo rays")

        #self.pool.map(shade, IC)
        for r in R:
            r[1].ell += r[0]

        res = []
        for i in I:
            if i is None:
                res.append([0., 0., 0.])
            else:
                res.append(i.ell*i.color)

        return res

def shade(sceneInterSampleM):
    sceneInterSampleM[1].ell += MonteCarlo(sceneInterSampleM[1], sceneInterSampleM[0], sceneInterSampleM[2])