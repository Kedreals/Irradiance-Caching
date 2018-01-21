# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:58:48 2017

@author: kedreals
"""

import numpy as np
from shape import Shape


class Plane(Shape):

    def __init__(self, pos, normal, ell=0.0, color=np.array([1., 1., 1.])):
        super().__init__(ell, color)

        self.pos = pos
        self.n = normal / np.linalg.norm(normal)

    def intersect(self, ray, intersection):

        o = np.dot(ray.d, self.n)

        if o * o < 0.001:
            return False

        t = np.dot(self.pos - ray.o, self.n) / np.dot(ray.d, self.n)

        if (t > 0) & (t < ray.t):
            ray.t = t
            intersection.pos = ray.o + ray.t * ray.d

            if np.dot(ray.d, self.n) > 0:
                intersection.n = -self.n
            else:
                intersection.n = self.n

            intersection.ell = self.ell
            intersection.color = self.color
            return True

        return False


class Rectangle(Plane):

    def __init__(self, pos, normal, bounds, ell=0., color=np.array([1., 1., 1.])):
        super().__init__(pos, normal, ell, color)

        self.bounds = bounds

    def intersect(self, ray, intersection):

        o = np.dot(ray.d, self.n)

        if o * o < 0.001:
            return False

        t = np.dot(self.pos - ray.o, self.n) / o

        if (t < 0) | (t >= ray.t):
            return False

        p = ray.o + t * ray.d
        o = np.zeros(2)
        o[0] = -np.arccos(self.n[2])
        o[1] = -np.arctan2(self.n[1], self.n[0])

        RT = np.array([[np.cos(o[0]), 0., np.sin(o[0])], [0., 1., 0.], [-np.sin(o[0]), 0., np.cos(o[0])]])
        RP = np.array([[np.cos(o[1]), np.sin(o[1]), 0.], [-np.sin(o[1]), np.cos(o[1]), 0.], [0., 0., 1.]])

        p = np.dot(RT, np.dot(RP, p - self.pos))

        if (abs(p[0]) > self.bounds[0]) | (abs(p[1]) > self.bounds[1]):
            return False

        ray.t = t
        intersection.pos = ray.o + ray.t * ray.d

        if np.dot(self.n, ray.d) > 0:
            intersection.n = -self.n
        else:
            intersection.n = self.n

        intersection.ell = self.ell
        intersection.color = self.color
        return True
