from integrator import Integrator
from irradiance_sample import *
import numpy as np
from intersection import Intersection
from ray import Ray
from multiprocessing import Pool
from Octree import Octree
import time

"""    def generateSample(self, intersection, scene, camera, ray, depth = 0):
        sample = Irradiance_Sample(intersection.pos, intersection.n)
        l = np.min([intersection.ell + self.MonteCarlo(intersection, scene, sample=sample), 1.])
        pixelSpacing = self.computeIntersectionPixelSpacing(camera, ray, intersection)
        sample.irradiance = l
        sample.computeSampleMaxContribution(self.minPixelDist, self.maxPixelDist, pixelSpacing)
        if self.showSamples :
            camera.image[ray.pixel[0], ray.pixel[1], :] = [1., 0., 0.]
        return sample
"""


class IrradianceIntegrator(Integrator):

    def __init__(self, minPixelDist, maxPixelDist, minWeight, maxCosAngleDiff, showSamplePoints=False,
                 maxBounceDepth=2):

        super().__init__()

        # pbrt 787
        self.minPixelDist = minPixelDist
        self.maxPixelDist = maxPixelDist
        self.minWeight = minWeight
        self.maxCosAngleDiff = maxCosAngleDiff
        # cache[0] are direct light samples, cache[1] are indirect light samples with one bounce
        # cache[2] are indirect light samples with two bounces
        self.showSamples = showSamplePoints
        self.maxBounceDepth = maxBounceDepth
        self.cache = []
        [self.cache.append(Octree([0.0, 0.0, 0.0], 50, [])) for i in range(maxBounceDepth + 1)]
        self.parallel = False

    def generateSample(self, intersection, scene, camera, ray, depth=0):
        sample = Irradiance_Sample(intersection.pos, intersection.n)
        minSamples = 64
        l = np.array([0., 0., 0.])

        # generate a sample for irradiance
        if (depth > 0):
            numSample = minSamples  # * depth
            for i in range(numSample):
                d = self.getCosineWeightedPointR3(intersection.n)
                r = Ray(intersection.pos + 0.001 * intersection.n, d)
                ni = Intersection()
                if (scene.intersect(r, ni)):
                    if (sample.minHitDist > r.t):
                        sample.minHitDist = r.t
                    procData = Irradiance_ProcessData(ni.pos, ni.n, self.minWeight, self.maxCosAngleDiff)
                    intVal = self.getInterpolatedValue(procData, depth - 1)
                    # if interpolation is successful
                    if (intVal >= 0).all():
                        lightval = ni.color * intVal * ni.BSDF(procData.avgLightDir, d, ni.n)
                        """
                        if lightval < 0:
                            print("\033[34mWARNING\033[30m: Interpolation lead to a negative light value")
                        """

                        l += lightval
                        sample.avgLightDir += d * np.linalg.norm(lightval)
                    # else make a new sample und use its irradiance
                    else:
                        s = self.generateSample(ni, scene, camera, r, depth - 1)
                        lightval = ni.color * s.irradiance * ni.BSDF(s.avgLightDir, d, ni.n)
                        if (lightval < 0).any():
                            print("\033[31mERROR\033[30m: Generating a new sample lead to a negative light value")
                        l += lightval
                        sample.avgLightDir += d * np.linalg.norm(lightval) #s.avgLightDir

                    if np.dot(sample.avgLightDir, sample.normal) < 0:
                        print("\033[34mWARNING\033[30m: The average Light direction points temporally in the wrong half space")

            l *= np.pi/numSample
        else:
            # generate a sample for direct light
            l = self.MonteCarlo(intersection, scene, minSamples, sample) #+ intersection.ell

        pixelSpacing = self.computeIntersectionPixelSpacing(camera, ray, intersection)
        sample.irradiance = l
        sample.computeSampleMaxContribution(self.minPixelDist, self.maxPixelDist, pixelSpacing)
        norm = np.linalg.norm(sample.avgLightDir)
        if (norm > 0):
            sample.avgLightDir /= norm

        if ((self.showSamples) & (depth == self.maxBounceDepth)):
            camera.image[ray.pixel[0], ray.pixel[1], :] = [1., 0., 0.]

        if np.dot(sample.avgLightDir, sample.normal) < 0:
            print("\033[31mERROR\033[30m: Sample was generated with avgLightDir pointing in the wrong half space")

        self.cache[depth].addObj(sample)
        return sample

    def fillCachParallelHelp(self, XYSC):
        ray = XYSC[3].generateRay(XYSC[0], XYSC[1])
        intersection = Intersection()
        if XYSC[2].intersect(ray, intersection):
            self.generateSample(intersection, XYSC[2], XYSC[3], ray, self.maxBounceDepth)

    def fillCache(self, camera, scene):
        pix_x = camera.image.shape[0]
        pix_y = camera.image.shape[1]

        print(pix_x)
        print(pix_y)
        print(self.maxPixelDist)

        if (self.parallel):
            p = Pool(8)
            A = [(x, y, scene, camera) for x in range(0, pix_x, self.maxPixelDist) for y in
                 range(0, pix_y, self.maxPixelDist)]
            p.map(self.fillCachParallelHelp, A)
        else:
            for i in range(0, pix_x, self.maxPixelDist):
                print("Filling Cache :",
                      int(10000 * (i / self.maxPixelDist) / (int(pix_x / self.maxPixelDist) + 1)) / 100, "%")
                for j in range(0, pix_y, self.maxPixelDist):
                    print(i, " ", j)
                    ray = camera.generateRay(i, j)
                    intersection = Intersection()
                    if (scene.intersect(ray, intersection)):
                        s = self.generateSample(intersection, scene, camera, ray, self.maxBounceDepth)

    # cosine weighted monte carlo integration over the hemisphere at the intersection
    def MonteCarlo(self, intersection, scene, sampleCount=64, sample=None):
        res = 0.0
        minHitDist = np.infty

        for i in range(sampleCount):
            d = self.getCosineWeightedPointR3(intersection.n)
            r = Ray(intersection.pos + 0.001 * intersection.n, d)
            ni = Intersection()
            if (scene.intersect(r, ni)):
                if (r.t < minHitDist):
                    minHitDist = r.t
                res += ni.ell * ni.color

                #if a sample is given, add the current hemisphere ray to average light direction
                #weight it by how much impact it has on the resulting radiance
                if ((sample != None) & (ni.ell > 0)):
                    sample.avgLightDir += d * ni.ell

        res *= np.pi / sampleCount
        if (sample != None):
            #normalize average light direction
            if (np.linalg.norm(sample.avgLightDir > 0)):
                sample.avgLightDir = sample.avgLightDir / np.linalg.norm(sample.avgLightDir)
            #min Hit distance is the closest intersection found while shooting rays in the hemisphere
            sample.minHitDist = minHitDist
        sample.irradiance = res #+ intersection.ell

        return res #+ intersection.ell

    def ell(self, scene, ray, camera):
        #very first call for ell() means the cache has to be filled
        if ((ray.pixel[0] == 0) & (ray.pixel[1] == 0)):
            start = time.perf_counter()
            self.fillCache(camera, scene)
            end = time.perf_counter()
            s = end - start
            m = int(s / 60)
            s = s % 60
            h = int(m / 60)
            m = m % 60

            print("filling cache took:", h, "h ", m, "min ", s, "s")
            for i in range(len(self.cache)):
                print("In cache depth: ", i, "are ", self.cache[i].objCount, "samples")
            #obsolete, because cannot iterate through octree
            #for i in range(len(self.cache)):
            #   ze = sum(s.irradiance == 0 for s in self.cache[i])
            #    print("In cache depth: ", i, " have ä", ze, " of ", len(self.cache[i]), "elements 0 (ir)radiance")

        intersection = Intersection()
        val = np.array([0.0, 0., 0.])
        if (scene.intersect(ray, intersection)):
            #interpolate indirect light
            for i in range(self.maxBounceDepth, 0, -1):
                interpolatedPoint = Irradiance_ProcessData(intersection.pos, intersection.n, self.minWeight,
                                                           self.maxCosAngleDiff)
                interpval = self.getInterpolatedValue(interpolatedPoint, i)
                e = 0
                if (interpval >= 0).all():
                    e += interpval * intersection.BSDF(interpolatedPoint.avgLightDir, -ray.d, intersection.n)

                #if interpolation failed, compute new sample
                else:
                    #print("new sample generated at ", ray.pixel)
                    s = self.generateSample(intersection, scene, camera, ray, i)
                    e += s.irradiance * intersection.BSDF(s.avgLightDir, -ray.d, s.normal)

                if (e < 0).any():
                    print(e)

                val += e

            #compute direct light
            sample = Irradiance_Sample(intersection.pos, intersection.n)
            self.MonteCarlo(intersection, scene, sampleCount=64, sample=sample)

            val += sample.irradiance * intersection.BSDF(sample.avgLightDir, -ray.d, intersection.n)
            if(val < 0).any():
                print("light value is negative :", val)

        return (val + intersection.ell) * intersection.color


    def getInterpolatedValue(self, interpolationPoint, depth):
        #interpolate all samples, that might be useful here
        for sample in self.cache[depth].find(interpolationPoint.pos):
            self.interpolate(interpolationPoint, sample)
        #if interpolated value error is below threshold, return value
        if (self.interpolationSuccessful(interpolationPoint)):
            norm = np.linalg.norm(interpolationPoint.avgLightDir)
            if (norm > 0):
                interpolationPoint.avgLightDir /= norm
            return interpolationPoint.irradiance / interpolationPoint.sumWeight
        #else return a sign, that this is not a useful interpolation
        else:
            return np.array([-1, -1, -1])

    #magic formula from pbrt book
    def computeIntersectionPixelSpacing(self, camera, ray, intersection):
        #2 rays one pixel in y and one in x direction from current ray
        rayx = camera.generateRay(ray.pixel[0] + 1, ray.pixel[1])
        rayy = camera.generateRay(ray.pixel[0], ray.pixel[1] + 1)

        #compute how much space in the world the current pixel overlaps
        #using differential geometry
        d = -np.dot(intersection.n, intersection.pos)
        tx = -(np.dot(intersection.n, rayx.o) + d) / np.dot(intersection.n, rayx.d)
        px = rayx.o + tx * rayx.d

        ty = -(np.dot(intersection.n, rayy.o) + d) / np.dot(intersection.n, rayy.d)
        py = rayy.o + ty * rayy.d

        dpdx = px - intersection.pos
        dpdy = py - intersection.pos

        pixelspacing = np.sqrt(np.linalg.norm(np.cross(dpdx, dpdy)))
        return pixelspacing

    #interpolate with error calulation used in pbrt book
    def interpolate(self, newPoint, sample):
        # pbrt 794
        perr = np.linalg.norm(newPoint.pos - sample.pos) / sample.maxDist
        val = np.max([0, (1.0 - np.dot(newPoint.normal, sample.normal)) / (1.0 - newPoint.maxCosAngleDiff)])
        if (val < 0):
            print(val)
            print(np.dot(newPoint.normal, sample.normal))
            print((1.0 - newPoint.maxCosAngleDiff))
        nerr = np.sqrt(val)

        err = np.max([perr, nerr])

        if (err < 1.0):
            weight = 1.0 - err
            newPoint.irradiance += weight * sample.irradiance
            newPoint.avgLightDir += weight * sample.avgLightDir
            newPoint.sumWeight += weight

        """
        if np.dot(newPoint.normal, newPoint.avgLightDir) < 0:
            print("Dam Dam DAAAAAAM")
        """
    def interpolationSuccessful(self, newPoint):
        # pbrt 794
        return newPoint.sumWeight >= newPoint.minWeight
