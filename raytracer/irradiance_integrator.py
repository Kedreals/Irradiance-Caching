from integrator import Integrator
from irradiance_sample import Irradiance_Sample as Irr_S
import numpy as np

class IrradianceIntegrator(Integrator):

    def __init__(self, minPixelDist, maxPixelDist, maxCosAngleDiff):

        super().__init__()

        self.minPixelDist = minPixelDist
        self.maxPixelDist = maxPixelDist
        self.maxCosAngleDiff = maxCosAngleDiff

    def ell(self, scene, ray):
        if (scene.intersect(ray)):
            return 1.0

        return 0.0

    def compute_Error(self, newPoint, sample):

        perr = (newPoint - sample.pos).length / sample.maxDist
        nerr = np.sqrt((1.0 - np.dot(newPoint.normal, sample.normal)) / (1.0 - self.maxCosAngleDiff))

        err = np.max(perr, nerr)

        if(err < 1.0):
            weight = 1.0 - err
            newPoint.irradiance += weight * sample.irradiance
            newPoint.avgLightDir += weight * sample.avgLightDir
            self.sumIrradiance
