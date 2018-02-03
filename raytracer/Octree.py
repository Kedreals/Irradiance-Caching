import numpy as np
import itertools as itter

MAX_OBJ_PER_LEAF = 100

def collide(tupel):
    return np.linalg.norm(tupel[0].pos - tupel[1].pos) <= tupel[0].maxDist + tupel[1].maxDist

class Octree:
    def __init__(self, position, size, objData):
        self.position = position
        self.size = size
        self.objData = objData
        self.branches = [None for i in range(8)]
        self.objCount = 0
        self.ObjPerLeaf = MAX_OBJ_PER_LEAF

    def isLeaf(self):
        return sum([1 for i in range(8) if self.branches[i] is not None]) == 0

    def collide(self, objData):
        size = self.size + objData.maxDist
        v = objData.pos - self.position
        return sum([1 for i in range(3) if abs(v[i]) <= size]) == 3

    def enlarge(self):
        self.ObjPerLeaf *= 2

    def split(self):
        check = list(itter.combinations(self.objData, 2))
        b = np.array(list(map(collide, check)))

        if b.all():
            self.enlarge()
            return

        for i in range(8):
            pos = (np.array([i % 2, int(i / 2) % 2, int(i / 4) % 2]) * 2 - np.ones(3)) * self.size / 2
            self.branches[i] = Octree(self.position + pos, self.size / 2, [])
            for obj in self.objData:
                if self.branches[i].collide(obj):
                    self.branches[i].addObj(obj)
        self.objData = []

    def addObj(self, objData):
        self.objCount += 1
        if self.isLeaf():
            self.objData.append(objData)
            # if is too much split
            if len(self.objData) >= self.ObjPerLeaf:
                self.split()
        else:
            for i in range(8):
                if self.branches[i].collide(objData):
                    self.branches[i].addObj(objData)

    def find(self, position):
        if self.isLeaf():
            return self.objData
        else:
            p = position - self.position
            b = 0
            for i in range(3):
                if p[i] > 0:
                    b += 2 ** i
            return self.branches[b].find(position)
