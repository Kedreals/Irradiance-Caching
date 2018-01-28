import numpy as np
from shape import Shape
from plane import Rectangle

class Cuboid(Shape):

    def __init__(self, pos, bounds = np.array([1.0, 1.0, 1.0]), rotation = np.array([0.0, 0.0, 0.0]), ell = 0.0, color = np.array([1.0, 1.0, 1.0])):
        super().__init__(ell, color)
        self.pos = pos
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]), 0], [np.sin(rotation[2]), np.cos(rotation[2]), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(rotation[1]), 0, np.sin(rotation[1])], [0, 1, 0], [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])
        Rx = np.array([[1, 0, 0], [0, np.cos(rotation[0]), -np.sin(rotation[0])], [0, np.sin(rotation[0]), np.cos(rotation[0])]])
        self.base = np.dot(np.dot(Rz, np.dot(Ry, Rx)), B) #still orthogonal base because rotation is orthogonal
        self.bounds = bounds

        """
        self.bounds = bounds
        self.rotation = rotation

        self.sides = [#upper
                      Rectangle(pos + np.array([-bounds[0], 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
                                np.array([bounds[2], bounds[1]]), ell, color),
                      #lower
                      Rectangle(pos + np.array([bounds[0], 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                                np.array([bounds[2], bounds[1]]), ell, color),
                      #left
                      Rectangle(pos + np.array([0.0, bounds[1], 0.0]), np.array([0.0, 1.0, 0.0]),
                                np.array([bounds[0], bounds[2]]), ell, color),
                      #right
                      Rectangle(pos + np.array([0.0, -bounds[1], 0.0]), np.array([0.0, -1.0, 0.0]),
                                np.array([bounds[0], bounds[2]]), ell, color),
                      #front
                      Rectangle(pos + np.array([0.0, 0.0, bounds[2]]), np.array([0.0, 0.0, 1.0]),
                                np.array([bounds[0], bounds[1]]), ell, color),
                      #back
                      Rectangle(pos + np.array([0.0, 0.0, -bounds[2]]), np.array([0.0, 0.0, -1.0]),
                                np.array([bounds[0], bounds[1]]), ell, color)
                      ]
        """

    def intersect(self, ray, intersection):
        p = ray.o-self.pos
        if (abs(np.dot(self.base, ray.d)) <= 0.001).all():
            return False
        #left and right side of the inequalities bounding the t
        L = -np.array([self.bounds[0], self.bounds[1], self.bounds[2]])
        R = np.array([self.bounds[0], self.bounds[1], self.bounds[2]])
        #an index for all the normal vectors that points in a different direction than the ray direction
        I = np.dot(self.base, ray.d) < 0
        #if the origin is outside the bounds in one dimention where the direction points away from the position there is no intersection
        if (np.dot(self.base, p)[~I] > self.bounds[~I]).any():
            return False

        J = abs(np.dot(self.base, ray.d)) < 0.001
        if (abs(np.dot(self.base, p)[J]) > self.bounds[J]).any():
            return False

        L[I] = L[I]*-1
        R[I] = R[I]*-1
        L = L-np.dot(self.base, p)
        R = R-np.dot(self.base, p)

        #a help vector for finding out where the direction is orthogonal to the base vectors
        help = np.dot(self.base, ray.d)
        L[abs(help) > 0.001] = L[abs(help) > 0.001] / help[abs(help) > 0.001]
        R[abs(help) > 0.001] = R[abs(help) > 0.001] / help[abs(help) > 0.001]

        #if the max value on the left side is higher than the min value on the right hand side, than there is no solution
        if L[abs(help) > 0.001].max() > R[abs(help) > 0.001].min():
            return False

        #else the t value is the smalest possible value. In other words
        t = L[abs(help) > 0.001].max()
        if t > ray.t or t <= 0:
            return False

        r = p+t*ray.d
        n = self.base[L == L[abs(help) > 0.001].max()][0]
        if np.dot(r, n) < 0:
            n = -n

        ray.t = t
        intersection.color = self.color
        intersection.ell = self.ell
        intersection.pos = ray.o + t*ray.d
        intersection.n = n

        return True

