from integrator import Integrator
import irradiance_sample
import numpy as np
from intersection import Intersection
from ray import Ray

class IrradianceIntegrator(Integrator):

    def __init__(self, minPixelDist, maxPixelDist, minWeight, maxCosAngleDiff):

        super().__init__()

        #pbrt 787
        self.minPixelDist = minPixelDist
        self.maxPixelDist = maxPixelDist
        self.minWeight = minWeight
        self.maxCosAngleDiff = maxCosAngleDiff

    def getCosineWeightedPointR3(self, n):
        fn = np.array([0., 0., 1.])

        o = np.zeros(2)
        o[0] = np.arccos(n[2])
        o[1] = np.arctan2(n[1], n[0])

        omega = np.random.rand(2)
        omega[0] = np.arcsin(np.sqrt(omega[0]))
        omega[1] *= np.pi*2

        RotTheta = np.array([[np.cos(o[0]), 0., np.sin(o[0])], [0., 1., 0.], [-np.sin(o[0]), 0., np.cos(o[0])]])
        RotPhi = np.array([[np.cos(o[1]), np.sin(o[1]), 0.], [-np.sin(o[1]), np.cos(o[1]), 0.], [0., 0., 1.]])

        r = np.zeros(3)
        r[0] = np.sin(omega[0])*np.cos(omega[1])
        r[1] = np.sin(omega[0])*np.sin(omega[1])
        r[2] = np.cos(omega[0])

        r = np.dot(RotPhi, np.dot(RotTheta, r))

        return r

    def MonteCarlo(self, intersection, scene, sampleCount = 64):
        res = 0.0

        for i in range(sampleCount):
            d = self.getCosineWeightedPointR3(intersection.n)
            r = Ray(intersection.pos+0.001*d, d)
            ni = Intersection()
            if(scene.intersect(r, ni)):
                res += ni.ell*np.pi

        res *= 1/sampleCount

        return np.min([res, 1.0])

    def ell(self, scene, ray):
        intersection = Intersection()
        if (scene.intersect(ray, intersection)):
            l = np.min([intersection.ell + self.MonteCarlo(intersection, scene), 1.])
            return l * intersection.color

        return np.zeros(3)

    def interpolate(self, newPoint, sample):
        #pbrt 794
        perr = np.linalg.norm(newPoint.pos - sample.pos) / sample.maxDist
        nerr = np.sqrt((1.0 - np.dot(newPoint.normal, sample.normal)) / (1.0 - newPoint.maxCosAngleDiff))

        err = np.max(perr, nerr)

        if(err < 1.0):
            weight = 1.0 - err
            newPoint.irradiance += weight * sample.irradiance
            newPoint.avgLightDir += weight * sample.avgLightDir
            newPoint.sumWeight += weight

    def interpolationSuccessful(self, newPoint):
        #pbrt 794
        return newPoint.sumWeight >= newPoint.minWeight
