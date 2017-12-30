from integrator import Integrator
import irradiance_sample
import numpy as np

class IrradianceIntegrator(Integrator):

    def __init__(self, minPixelDist, maxPixelDist, minWeight, maxCosAngleDiff):

        super().__init__()

        self.minPixelDist = minPixelDist
        self.maxPixelDist = maxPixelDist
        self.minWeight = minWeight
        self.maxCosAngleDiff = maxCosAngleDiff

    def ell(self, scene, ray):
        if (scene.intersect(ray)):
            return 1.0

        return 0.0

    def computeError(self, newPoint, sample):

        perr = np.linalg.norm(newPoint.pos - sample.pos) / sample.maxDist
        nerr = np.sqrt((1.0 - np.dot(newPoint.normal, sample.normal)) / (1.0 - newPoint.maxCosAngleDiff))

        err = np.max(perr, nerr)

        if(err < 1.0):
            weight = 1.0 - err
            newPoint.irradiance += weight * sample.irradiance
            newPoint.avgLightDir += weight * sample.avgLightDir
            newPoint.sumWeight += weight

    def interpolationSuccessful(self, newPoint):
        return newPoint.sumWeight >= newPoint.minWeight