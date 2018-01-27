import numpy as np

class Irradiance_Sample():

    def computeSampleMaxContribution(self, minPixelSpacing, maxPixelSpacing, intersectionPixelSpacing):
        #pbrt 797
        minDist = minPixelSpacing * intersectionPixelSpacing
        maxDist = maxPixelSpacing * intersectionPixelSpacing
        contribExtend = np.clip(self.minHitDist / 2.0, minDist, maxDist)
        if contribExtend <= 0:
            print("\033[34mWARNING\033[30m: maxDist of sample is lower or equal to 0")
        self.maxDist = contribExtend
        return contribExtend


    def __init__(self, pos, normal, spectrum = 0, maxDistance = 0):
        #pbrt 793
        self.pos = pos
        self.normal = normal
        self.irradiance = spectrum
        self.minHitDist = np.infty
        self.maxDist = maxDistance
        self.avgLightDir = np.zeros(3)
        self.rotGrad = np.zeros(3)

class Irradiance_ProcessData():
    def __init__(self, pos, normal, minWeight, maxCosAngleDiff):
        #pbrt 793
        self.pos = pos
        self.normal = normal
        self.minWeight = minWeight
        self.maxCosAngleDiff = maxCosAngleDiff
        self.sumWeight = 0.0
        self.samplesFound = 0.0
        self.irradiance = 0.0
        self.avgLightDir = np.array([0.0, 0.0, 0.0])