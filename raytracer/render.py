#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:47:48 2017

@author: lessig
"""

import numpy as np
import matplotlib.pyplot as plt

from ray import Ray
from camera import Camera
from sphere import Sphere
from plane import Plane
from plane import Rectangle
from cuboid import Cuboid
from scene import Scene
from basic_integrator import BasicIntegrator
from irradiance_integrator import IrradianceIntegrator
import time
from multiprocessing import Pool

def createScene(name="simple"):
    scene = Scene()

    if name == "simple":
        plane = Rectangle(np.array([1., 0, 3.0]), np.array([-1., 0., 0.]), 0.75 * np.array([1, 1]), 1)
        sphere = Sphere(np.array([0.0, 0.0, 3.0]), 1.0)

        scene.objects.append(sphere)
        scene.objects.append(plane)
    elif name == "cornell":
        size = 4
        pos = 3
        leftWall = Rectangle(pos=np.array([0, -size, pos]), normal=np.array([0, 1, 0]), bounds=np.array([size, size]),
                             color=np.array([1., 0.2, 0.2]))
        rightWall = Rectangle(pos=np.array([0, size, pos]), normal=np.array([0, -1, 0]), bounds=np.array([size, size]),
                              color=np.array([0.2, 1., 0.2]))
        floor = Rectangle(pos=np.array([size, 0, pos]), normal=np.array([-1, 0, 0]), bounds=np.array([size, size]),
                          color=np.array([1., 1., 1.]))
        ceiling = Rectangle(pos=np.array([-size, 0, pos]), normal=np.array([1, 0, 0]), bounds=np.array([size, size]),
                            color=np.array([1., 1., 1.]))
        front = Rectangle(pos=np.array([0, 0, pos-size]), normal=np.array([0, 0, 1]), bounds=np.array([size, size]),
                          color=np.array([1., 1., 1.]))
        back = Rectangle(pos=np.array([0, 0, pos+size]), normal=np.array([0, 0, -1]), bounds=np.array([size, size]),
                         color=np.array([1., 1., 1.]))
        light = Rectangle(pos=np.array([-size + 0.001, 0, pos+size/2]), normal=np.array([1, 0, 0]), bounds=np.array([0.5, 1]),
                          color=np.array([1., 1., 0.7]), ell=10)

        quader = Cuboid(pos=np.array([1.5, -1.25, pos+size-1.5]), bounds=np.array([2.5, 1, 1]), rotation=np.array([-35.*np.pi/180., 0, 0]))
        cube = Cuboid(pos=np.array([3, 1.25, pos+size-2.5]), bounds=np.array([1., 1., 1.]), rotation=np.array([35.*np.pi/180, 0, 0]))

        scene.objects.append(leftWall)
        scene.objects.append(rightWall)
        scene.objects.append(floor)
        scene.objects.append(ceiling)
        scene.objects.append(front)
        scene.objects.append(back)
        scene.objects.append(light)
        scene.objects.append(quader)
        scene.objects.append(cube)
    elif name == "box":

        size = 4
        pos = 3


        leftWall = Rectangle(np.array([0., -size, pos]), np.array([0., 1., 0.]), np.array([size, size]), color=np.array([1., 0.2, 0.2]))
        rightWall = Rectangle(np.array([0., size, pos]), np.array([0., -1., 0]), np.array([size, size]), color=np.array([0.2, 1., 0.2]))
        floor = Rectangle(np.array([size, 0., pos]), np.array([-1., 0., 0.]), np.array([size, size]), color=np.array([0.2, 0.2, 1.]))
        ceiling = Rectangle(np.array([-size, 0, pos]), np.array([1., 0., 0.]), np.array([size, size]), color=np.array([1., 1., 0.2]))
        back = Rectangle(np.array([0., 0., pos+size]), np.array([0., 0., -1.]), np.array([size, size]), color=np.array([1., 0.2, 1.]))
        front = Rectangle(np.array([0, 0., pos-size]), np.array([0., 0., 1.]), np.array([size, size]), color=np.array([1., 1., 1.]))
        squareLight = Rectangle(np.array([-size + 0.01, 0, pos]), np.array([1, 0, 0.]), np.array([size, size/2]), 3)
        redBall = Sphere(np.array([3., 0., pos+size-1]), 1, 0, np.array([1., 0.2, 1.]))
        cube = Cuboid(np.array([0., 3., pos+size-1]), np.array([3., 1., 1.]), rotation=np.array([0., 0., 0.]))

        scene.objects.append(leftWall)
        scene.objects.append(rightWall)
        scene.objects.append(floor)
        scene.objects.append(ceiling)
        scene.objects.append(back)
        scene.objects.append(front)
        scene.objects.append(squareLight)
        scene.objects.append(redBall)
        scene.objects.append(cube)

    return scene


def GenRay(i, res_y, camera):
    i_y = i % res_y
    i_x = int(i / res_y)

    return camera.generateRay(i_x, i_y)


def Lamb(x):
    return x[0].ell(x[1], x[2], x[3])


def renderList(res_x, res_y, scene, integrator):
    cam = Camera(res_x, res_y)

    N = res_x * res_y
    Rays = [(integrator, scene, GenRay(i, res_y, cam), cam) for i in range(N)]
    p = Pool(processes=8)
    ellvals = p.map(Lamb, Rays)

    for ix in range(res_x):
        for iy in range(res_y):
            cam.image[ix, iy, :] = ellvals[ix * res_y + iy]

    return cam.image


def render(res_x, res_y, scene, integrator):
    cam = Camera(res_x, res_y)

    for ix in range(res_x):
        d = int(10000 * ix / res_x) / 100
        print("\033[30mFinished Rendering\033[36m", d, "%\033[30m of the Pixels")
        for iy in range(res_y):

            r = cam.generateRay(ix, iy)

            ellval = integrator.ell(scene, r, cam)
            if (integrator.showSamples is not True) | ((cam.image[ix, iy, :] != 0).sum() == 0):
                cam.image[ix, iy, :] = ellval

    for i in range(len(integrator.cache)):
        print("\033[32mINFO:\033[30m Cache Level", i, "has", integrator.cache[i].objCount,"many samples")

    return cam.image


def renderTest(res_x, res_y, scene, integrator):
    cam = Camera(res_x, res_y)
    Rays = []
    for ix in range(res_x):
        for iy in range(res_y):
            Rays.append(cam.generateRay(ix, iy))

    I = integrator.ell2(scene, Rays, 256)
    for ix in range(res_x):
        for iy in range(res_y):
            cam.image[ix, iy, :] = I[ix * res_y + iy]

    return cam.image


def ScaleImageSqrt(image):
    image = np.sqrt(image)
    return image / image.max()


def ScaleImageLog(image):
    image = np.log10(image + 1)
    return image / image.max()


integrator = IrradianceIntegrator(1, 40, 0.1, np.pi / 4.0, False, 2, renderDirectLight=False, fillCache=True,
                                  directLightSampleCount=64)
scene = createScene("cornell")

resolution = 64

start = time.perf_counter()
im = render(resolution, resolution, scene, integrator)
end = time.perf_counter()

seconds = end - start
minutes = int(seconds / 60)
seconds = seconds % 60
hours = int(minutes / 60)
minutes = minutes % 60

print("Execution time = ", hours, ":", minutes, ":", seconds)

#plt.figure()
#plt.imshow(im)

plt.figure()
plt.imshow(ScaleImageSqrt(im))
plt.title("Square root scaled image")

plt.figure()
plt.imshow(ScaleImageLog(im))
plt.title("Logarithmically scaled image ")
plt.show()

