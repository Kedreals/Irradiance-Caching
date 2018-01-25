import numpy as np
from shape import Shape
from plane import Rectangle

class Cuboid(Shape):

    def __init__(self, pos, bounds = np.array([1.0, 1.0, 1.0]), rotation = np.array([0.0, 0.0, 0.0]), ell = 0.0, color = np.array([1.0, 1.0, 1.0])):
        super().__init__(ell, color)

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

    def intersect(self, ray, intersection):
        intersects = False
        possibleSides = []

        for i in range(3):
            if(np.dot(self.sides[i*2].n, ray.d) < 0):
                possibleSides.append(2*i+1)
            else:
                possibleSides.append(2*i)

        for i in possibleSides:
            intersects |= self.sides[i].intersect(ray, intersection)

        """
        for i in range(6):
            intersects |= self.sides[i].intersect(ray, intersection)
        """
        return intersects
