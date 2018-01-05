import numpy as np

class Irradiance_Sample():

    @staticmethod
    def computeSampleMaxContribution(minHitDist, minPixelSpacing, maxPixelSpacing, intersectionPixelSpacing):
        #pbrt 797
        minDist = minPixelSpacing * intersectionPixelSpacing
        maxDist = maxPixelSpacing * intersectionPixelSpacing
        contribExtend = np.clip(minHitDist, minDist, maxDist)
        return contribExtend


    def __init__(self, pos, normal, spectrum, maxDistance, avgLightDir):
        #pbrt 793
        self.pos = pos
        self.normal = normal
        self.irradiance = spectrum
        self.maxDist = maxDistance
        self.avgLightDir = avgLightDir

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