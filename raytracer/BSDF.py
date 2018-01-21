import numpy as np

def constBSDF(incomingLightDir, outgoingLightDir, normal):
    return 1

def DiffuseBSDF(incomingLightDir, outgoingLightDir, normal):
    return 1/np.pi#np.dot(incomingLightDir, normal)
