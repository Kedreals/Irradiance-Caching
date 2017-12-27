
from itegrator import Integrator

from integrator import Integrator


class IrradianceIntegrator(Integrator):

    def ell(self, scene, ray):
        if (scene.intersect(ray)):
            return 1.0

        return 0.0


