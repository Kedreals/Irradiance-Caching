# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:58:48 2017

@author: kedreals
"""

import numpy as np
from shape import Shape

class Plane( Shape):

    def __init__(self, pos, normal):
        super().__init__()

        self.pos = pos
        self.n = normal/np.linalg.norm(normal)

    def intersect(self, ray, intersection):

        o = np.dot(ray.d, self.n)

        if o*o < 0.001 :
            return False

        t = np.dot(self.pos - ray.o, self.n) / np.dot(ray.d, self.n)

        if (t > 0) & (t < ray.t):
            ray.t = t
            intersection.pos = ray.o + ray.t*ray.d
            intersection.n = self.n
            intersection.ell = self.ell
            return True

        return False

