#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:47:48 2017

@author: lessig
"""

from ray import Ray
from camera import Camera
from sphere import Sphere
from scene import Scene
from basic_integrator import BasicIntegrator
import numpy as np

def createScene() :
    
    scene = Scene()
    
    sphere = Sphere( np.array([0.0, 3.0, 0.0]), 1.0)
    scene.objects.append( sphere)
    
    return scene


def render( res_x, res_y, scene, integrator) :
    
    cam = Camera( res_x, res_y)
    
    for ix in range( res_x) :
        for iy in range( res_y) :
            
            r = cam.generateRay( ix, iy)
            
            ellval = integrator.ell( scene, r)
            cam.image[ix,iy] = ellval
            
    return cam.image
    


integrator = BasicIntegrator()
scene = createScene()

im = render( 512, 512, scene, integrator)

            