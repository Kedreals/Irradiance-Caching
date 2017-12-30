
class Irradiance_Sample():

    def __init__(self, pos, normal, spectrum, maxDistance, avgLightDir):

        self.pos = pos
        self.normal = normal
        self.irradiance = spectrum
        self.maxDist = maxDistance
        self.avgLightDir = avgLightDir