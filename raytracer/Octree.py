# This defines the maximum objects an LeafNode can hold, before it gets subdivided again.
MAX_OBJECTS_PER_CUBE = 10

# This dictionary is used by the findBranch function, to return the correct branch index
DIRLOOKUP = {"3":0, "2":1, "-2":2, "-1":3, "1":4, "0":5, "-4":6, "-3":7}


class OctNode :
    def __init__(self, position, size, data):
        self.position = position
        self.size = size
        self.isLeafNode = True
        self.data = data
        self.branches = [None, None, None, None, None, None, None, None]

class Octree:
    def __init__(self, worldSize):
        self.root = self.addNode((0,0,0), worldSize, [])
        self.worldSize = worldSize

    def addNode(self, position, size, objects):
        return OctNode(position, size, objects)

    def insertNode(self, root, size, parent, objData):
        if root == None:
            pos = parent.position
            offset = size/2
            branch = self.findBranch(parent, objData.position)
            newCenter = (0,0,0)
            if branch == 0:
                # left down back
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] - offset)

            elif branch == 1:
                # left down forwards
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] + offset)

            elif branch == 2:
                # right down forwards
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] + offset)

            elif branch == 3:
                # right down back
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] - offset)

            elif branch == 4:
                # left up back
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] - offset)

            elif branch == 5:
                # left up forward
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] + offset)

            elif branch == 6:
                # right up forward
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] - offset)

            elif branch == 7:
                # right up back
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] - offset)

            return self.addNode(newCenter, size, [objData])

            # else: are we not at our position, but not at a leaf node either
        elif root.position != objData.position and root.isLeafNode == False:

            # we're in an octNode still, we need to traverse further
            branch = self.findBranch(root, objData.position)
            # Find the new scale we working with
            newSize = root.size / 2
            # Perform the same operation on the appropriate branch recursively
            root.branches[branch] = self.insertNode(root.branches[branch], newSize, root, objData)
        # else, is this node a leaf node with objects already in it?
        elif root.isLeafNode:
            # We've reached a leaf node. This has no branches yet, but does hold
            # some objects, at the moment, this has to be less objects than MAX_OBJECTS_PER_CUBE
            # otherwise this would not be a leafNode (elementary my dear watson).
            # if we add the node to this branch will we be over the limit?
            if len(root.data) < MAX_OBJECTS_PER_CUBE:
                # No? then Add to the Node's list of objects and we're done
                root.data.append(objData)
                # return root
            elif len(root.data) == MAX_OBJECTS_PER_CUBE:
                # Adding this object to this leaf takes us over the limit
                # So we have to subdivide the leaf and redistribute the objects
                # on the new children.
                # Add the new object to pre-existing list
                root.data.append(objData)
                # copy the list
                objList = root.data
                # Clear this node's data
                root.data = None
                # Its not a leaf node anymore
                root.isLeafNode = False
                # Calculate the size of the new children
                newSize = root.size / 2
                # distribute the objects on the new tree
                # print "Subdividing Node sized at: " + str(root.size) + " at " + str(root.position)
                for ob in objList:
                    branch = self.findBranch(root, ob.position)
                    root.branches[branch] = self.insertNode(root.branches[branch], newSize, root, ob)
        return root

    def findPosition(self, root, position):
        # Basic collision lookup that finds the leaf node containing the specified position
        # Returns the child objects of the leaf, or None if the leaf is empty or none
        if root == None:
            return None
        elif root.isLeafNode:
            return root.data
        else:
            branch = self.findBranch(root, position)
            return self.findPosition(root.branches[branch], position)

    def findBranch(self, root, position):
        # helper function
        # returns an index corresponding to a branch
        # pointing in the direction we want to go
        vec1 = root.position
        vec2 = position
        result = 0
        # Equation created by adding nodes with known branch directions
        # into the tree, and comparing results.
        # See DIRLOOKUP above for the corresponding return values and branch indices
        for i in range(3):
            if vec1[i] <= vec2[i]:
                result += (-4 / (i + 1) / 2)
            else:
                result += (4 / (i + 1) / 2)
        result = DIRLOOKUP[str(result)]
        return result