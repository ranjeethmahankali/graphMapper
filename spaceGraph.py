import uuid
import math
import copy

#Handling the error message
def reportError(errMsg):
    print('Error: ' + errMsg)

#Connects the two spaces back and forth
def connectSpaces(space1, space2):
    c1 = space1.connect(space2)
    c2 = space2.connect(space1)

    return [c1,c2]

#this set contains all the parent-less root spaces in the document
roots = set()
#this is the dictionary of all Spaces in the document with their ids as keys
spaceList = dict()
#this is the dictionary of all connection objects with their ids as keys
connectionList = dict()

#returns the labels of all the root spaces in the document as a list
def rootLabels():
    rootArr = []
    for sp in roots:
        rootArr.append(sp.label)

    return rootArr

#this method prints all the spaces in the document created till now with their
#appropriate family structure
def printAllRoots():
    for space in roots:
        space.printSpace()

#prints the list of all spaces in the document with their ids
def printAllSpaces():
    for spID in spaceList:
        print(spaceList[spID].label + ' : ' + str(spID))

#returns the space with the given id from the set of all spaces in the document.
#returns none of no space exists with that id
def spaceWithID(spID):
    if spID in spaceList:
        return spaceList[spID]
    else:
        reportError('cannot find a space with the id : '+str(spID))

#This is the class for cSpace object
class cSpace:
    #class initialization
    def __init__(self, name, parentSpace = None):
        self.label = name
        #this is the unique id that is assigned to each space in order to avoid
        #confusion in cases where same label is used multiple times across a family tree
        #for example, the label 'toilet' could be used multiple times in a tree representing
        #an apartment building with many flats, but all toilets will have unique ids
        self.id = uuid.uuid4()
        if self.id in spaceList:
            reportError('Id clash while creating new space !!!')
        else:
            spaceList[self.id] = self
        #this is a dictionary that contains the children of this space
        self.c = dict()
        #this is the set that contains the references to the sibling spaces that are
        #connected to this space
        self.connected = set()
        #this is a dictionary of all incoming connection objects
        self.con = dict()
        #this dictionary contains terminal spaces with connection ids as keys
        self.t = dict()
        #this is the deciding the default value for the parent attribute
        #and also updating the roots of the document
        if parentSpace is None:
            self.parent = None
            roots.add(self)
        else:
            parentSpace.addChild(self)

    #This function adds the given space as a child to this space
    def addChild(self,space):
        labelIsValid = (not space.label in self.c) and self.label != space.label
        labelIsValid = labelIsValid and (not self.hasAncestor(space.label))
        if labelIsValid:
            space.parent = self
            self.c[space.label] = space
            #now removving the child space from the roots set if it was already there
            roots.discard(space)
        else:
            errMsg = 'space name '+space.label+' already taken by a sibling or an ancestor'
            reportError(errMsg)

    #This function adds multiple childrfen supplied as a list or a tuple
    def addChildren(self, spaceArr, isOpen = False):
        # isOpen param when set true will make all the children interconnected, like an open space with
        # zones rather than collection of rooms
        newL = []
        for sp in spaceArr:
            self.addChild(sp)
            newL.append(sp.label)
            
        if isOpen:
            l=0
            while l < len(newL)-1:
                j = l+1
                while j < len(newL):
                    connectSpaces(self.c[newL[l]], self.c[newL[j]])
                    j += 1
                l += 1

    #change the label of the space properly without messing up the family structu
    def changeLabel(self, newLabel):
        oldLabel = self.label
        self.label = newLabel
        del self.parent.c[oldLabel]
        self.parent.c[newLabel] = self

    #Adds a one-way connection from the given space to this space 
    def connect(self, space):
        if isinstance(space,cSpace):            
            if self.isSibling(space):
                self.connected.add(space)
                self.con[space.id] = connection(space,self)
                return self.con[space.id]
            else:
                errMsg = 'Cannot connect '+self.label+' to a nonSibling '+space.label
                reportError(errMsg)
                return None
        else:
            reportError('This is not a cSpace object so cannot connect it to anything')
            return None

    #this method adds terminalSpace as the terminal for this space for its connection with
    #the connectedSpace
    def addTerminal(self, conObj, terminalSpace):
        if terminalSpace.parent != self:
            reportError('Cannot add terminal')
            return None
        
        sp1 = conObj.fromSpace
        sp2 = conObj.toSpace

        isValid = False

        if self == sp2:
            isValid = True
        else:
            if self.parent:
                if conObj.id in self.parent.t and self.parent.t[conObj.id] == self:
                    isValid = True

        if isValid:
            self.t[conObj.id] = terminalSpace

    #returns the terminal space of this space for the connection conObj
    def terminal(self, conObj):
        if conObj.id in self.t:
            return self.t[conObj.id]
        else:
            #reportError('this terminal does not exist for ' + self.label)
            return None

    def accessTo(self, space):
        if space.id in self.con:
            return self.con[space.id]
        else:
            return None
        

    #This adds a two way conneciton between the given two children of this space
    def connectChildren(self, spaceLabel1, spaceLabel2):
        if self.hasChild(spaceLabel1) and self.hasChild(spaceLabel2):
            c = connectSpaces(self.c[spaceLabel1], self.c[spaceLabel2])
    
    #Returns a boolean if this space has the given space as a child
    def hasChild(self, spaceLabel):
        if spaceLabel in self.c:
            return True
        else:
            return False

    #returns true if this space is connected to the given space
    def isConnected(self, space):
        if space in self.connected:
            return True
        else:
            return False

    # instead of accessing directly through the array,
    # this method checks if the child with that label exists and then returns it
    def child(self, childLabel):
        if childLabel in self.c:
            return self.c[childLabel]
        else:
            reportError('No child with that label')
            return None

    #returns a list containing the children of this space
    def children(self):
        spaceList = []
        for sp in self.c:
            spaceList.append(sp)

        return spaceList

    #this method returns the root space of the family tree of which this space is a part of
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    #this method returns the depth of this space from its root
    def depth(self):
        if self.parent:
            return (1 + self.parent.depth())
        else:
            return 1

    #this method returns the relative depth of this space w.r.t another one
    def relDepth(self, space):
        return (space.depth() - self.depth())

    #removes the child space with the given label
    def removeChild(self,spaceLabel):
        if spaceLabel in self.c:
            del self.c[spaceLabel]

    #removes all the children spaces with labels in the provided list
    def removeChildren(self, spLabelArr):
        for spL in spLabelArr:
            self.removeChild(spL)

    #returns true if this space has a sibling with the given label
    def hasSibling(self, sibLabel):
        if not self.parent is None:
            if sibLabel in self.parent.c and self.label != sibLabel:
                return True
            elif self.label == sibLabel:
                err = 'the cSpace '+self.label+' is being tested if it is its own sibling'
                #reportError(err)
                return True
            else:
                return False
        else:
            reportError(self.label+' has no parents')
            return False

    #returns true if the given space is a sibling of this space
    def isSibling(self, sibSpace):
        if not self.parent is None:
            if sibSpace.label in self.parent.c and self.label != sibSpace.label:
                if self.parent.c[sibSpace.label].id == sibSpace.id:
                    return True
            elif self.label == sibSpace.label:
                err = 'the cSpace '+self.label+' is being tested if it is its own sibling'
                #reportError(err)
                return True
            else:
                return False
        else:
            reportError(self.label+' has no parents')
            return False

    #returns the sibling space object by reference
    def sibling(self, sibLabel):
        if self.hasSibling(sibLabel):
            return self.parent.c[sibLabel]
        else:
            reportError('Could not find Sibling')
            return None

    #this function checks if the cSpace has any descendant with label desLabel
    def hasDescendant(self, desLabel):
        if desLabel in self.c:
            return True
        
        for cL in self.c:
            if self.c[cL].hasDescendant(desLabel):
                return True

        return False

    #this function checks if the cSpace has any ancestor with the label anLabel
    def hasAncestor(self, anLabel):
        if not self.parent is None:
            if self.parent.label == anLabel:
                return True
        else:
            return False
            
        if self.parent.hasAncestor(anLabel):
            return True

        return False
    
    # this funtion prints the internal structure and children upto
    # maxDepth number of generations
    def printSpace(self, maxDepth = math.inf, d = 0):
        l = self.label
        tab = d*'\t'
        connections = ''
        if len(self.connected) == 0:
            connections = '-'
        else:
            count = 1
            for space in self.connected:
                name = space.label
                if count < len(self.connected):
                    name += ', '
                
                connections += name
                count += 1

        connections = '{' + connections + '}'

        spacePrint = tab + l + ': ' + connections
        print(spacePrint)
        d += 1

        if d <= maxDepth and len(self.c) > 0:
            for spC in self.c:
                self.c[spC].printSpace(maxDepth, d)

    #this method finds a path to the given sibling space
    #finds a path to the sibling with label sibLabel
    def findPathTo(self, targetSpace, path = []):
        if self.isSibling(targetSpace):
            #find the path
            path = path + [self]

            if self.label == targetSpace.label:
                return path

            for sp in self.connected:
                if sp not in path:
                    newPath = sp.findPathTo(targetSpace, path)
                    if newPath: return newPath
            return None
        else:
            err = 'cannot find path between non-siblings'
            reportError(err)
            return None

    #finds all paths to the sibling with label sibLabel
    def findAllPathsTo(self, targetSpace, path = []):
        if self.isSibling(targetSpace):
            #find all paths
            path = path + [self]

            if self.label == targetSpace.label:
                return [path]

            paths = []
            for sp in self.connected:
                if sp not in path:
                    newPaths = sp.findAllPathsTo(targetSpace, path)
                    for newPath in newPaths:
                        paths.append(newPath)

            return paths
        else:
            err = 'cannot find any paths between non-siblings'
            reportError(err)
            return None

    #finds the shortest path to the sibling with label sibLabel
    def findShortestPathTo(self, targetSpace, path = []):
        if self.isSibling(targetSpace):
            #find the shortest path
            path = path + [self]

            if self.label == targetSpace.label:
                return path

            shortestPath = None
            for sp in self.connected:
                if sp not in path:
                    newPath = sp.findShortestPathTo(targetSpace, path)
                    if newPath:
                        if not shortestPath or len(shortestPath) > len(newPath):
                            shortestPath = newPath

            return shortestPath
        else:
            err = 'cannot find a path between non-siblings'
            reportError(err)
            return None

    # this method returns the path through the family tree from this space and a
    # non-sibling space
    def relationTo(self, space, path = [], direction = 0):
        # direction 0 for the node that is the starting node
        # direction -1 for node that is reached by travelling up the family tree
        # direction 1 for node that is reached by travelling down the family tree
        path = path + [[self, direction]]

        if self.id == space.id:
            return path

        # all() and any() are essentially like logic gates for more than two booleans at once
        # all booleans can be passed iterably directly in the parantheses with a
        # syntax similar to that of a for loop
        if (not self.parent is None) and all(self.parent not in pE for pE in path):
            newPath = self.parent.relationTo(space, path, -1)
            if newPath: return newPath

        for ch in self.c:
            if all(self.c[ch] not in pE for pE in path):
                newPath = self.c[ch].relationTo(space, path, 1)
                if newPath: return newPath

        return None

    #this function clones a given space with all the same attributes except the label
    #which is provided as a parameter. The clone will have same connections to the siblings
    #the clone will also occupy the same position in the family tree - like a twin brother
    #to the original space
    #But if you specifiy a parentSpace then the cloned space will be a child of that
    #and it will have no connections
    def clone(self, newLabel, parentSpace = 'sameParent'):
        if parentSpace == 'sameParent':parentSpace = self.parent

        newSpace = copy.deepcopy(self)
        newSpace.label = newLabel
        newSpace.connected = set()

        if parentSpace:
            parentSpace.addChild(newSpace)
            if parentSpace is self.parent:
                for space in self.connected:
                    connectSpaces(newSpace, space)
        else:
            newSpace.parent = parentSpace#and here parentSpace is None
            roots.add(newSpace)

        #add newSpace and all the descendants of the newSpace to the spaceList
        def addSpace(space):
            #giving the space a unique id because this is a copied object
            space.id = uuid.uuid4()
            if space.id in spaceList:
                reportError('Id clash while cloning ' + self.label + ' !!!')
                return None
            else:
                spaceList[space.id] = space

                for spL in space.c:
                    addSpace(space.c[spL])

        addSpace(newSpace)

        return newSpace

    #this function navigates to any space that is under the same root
    def navigateTo(self, targetSpace):
        selfRoot = self.root()
        targetRoot = targetSpace.root()

        if selfRoot.id == targetRoot.id:
            #navigation starts
            route = []
            rel = self.relationTo(targetSpace)

            i = 0
            c = None #indec of the common ancestor in the list
            p1 = None #index of parent 1 in the list
            p2 = None #index of parent 2 in the list
            while i < len(rel)-1:
                #print(rel[i+1][1])
                if rel[i+1][1] == 1:
                    c = i
                    if i != 0:
                        p1 = i-1
                        p2 = i+1
                    break

                i += 1

            #print(c,p1,p2)

            if c and p1 and p2:
                #print(rel[p1][0].label,'and', rel[p2][0].label)
                #this is the list up to the child of the common ancestor
                rel1 = rel[:(p1+1)]
                #this is the path b/n children of the common ancestor
                path = rel[p1][0].findShortestPathTo(rel[p2][0])
                #this is the path from the child of the common ancestor
                rel2 = rel[p2:]

                i = 0
                while i < len(rel1):
                    direction = None
                    
                    if i == 0:
                        direction = 'start'
                    else:
                        direction = -1

                    route.append([rel1[i][0], direction])
                    
                    i += 1

                i = 1
                while i < len(path):
                    route.append([path[i],0])
                    i += 1

                i = 1
                while i < len(rel2):
                    route.append([rel2[i][0],1])
                    i += 1
                    
            else:
                i = 0
                while i < len(rel):
                    direction = None
                    if i == 0:
                        direction = 'start'
                    else:
                        direction = rel[i][1]
                        
                    route.append([rel[i][0],direction])

                    i += 1

            #processing for navigation through terminal spaces - starts
            #while leaving the source space
            i = 0
            while i < len(route)-1:
                if route[i+1][1] == 0 and route[i][1] == -1:
                    j = i
                    while j > 0:
                        if route[j+1][1] == 0 and route[j][1] == -1:
                            access = route[j][0].accessTo(route[j+1][0])
                            terminalSpace = route[j][0].terminal(access)
                            if terminalSpace:
                                traverse = route[j-1][0].findShortestPathTo(terminalSpace)

                                k = 1
                                while k < len(traverse):
                                    route.insert(j+k-1, [traverse[k], 0])
                                    k += 1

                                j += len(traverse)-1
                        
                        j -= 1

                elif route[i+1][1] == 1 and route[i][1] == 0:
                    access = route[i][0].accessTo(route[i-1][0])
                    terminalSpace = route[i][0].terminal(access)

                    if terminalSpace:
                        traverse = terminalSpace.findShortestPathTo(route[i+1][0])
                        del route[i+1]

                        k = 0
                        while k < len(traverse):
                            direction = 0
                            if k == 0:
                                direction = 1
                            
                            route.insert(i+k+1, [traverse[k], direction])
                            k += 1

                i += 1
                
            #processing for navigation through terminal spaces - ends
            return route

        else:
            errMsg = 'Cannot navigate because '+self.label+' and '+targetSpace.label+' do not share a common root'
            reportError(errMsg)
            return []

# returns the path as a string with spaces separated by arrows
def printPath(path):
    pathPrint = ''
    
    i = 0
    while i < len(path):
        pathPrint += path[i].label
        
        if i != len(path)-1:
            pathPrint += ' -> '

        i += 1
            
    return pathPrint

# returns the relation as a string with parents and children denoted by
# closing and opening braces respectively
def printRelation(relation):
    relPrint = ''
    l = len(relation)

    if l == 1:
        return relation[0][0].label

    i = 0
    while i < l-1:
        if relation[i+1][1] == 1:
            relPrint += relation[i][0].label + ' ( '
        elif relation[i+1][1] == -1:
            relPrint += relation[i][0].label + ' ) '

        i += 1

    relPrint += relation[l-1][0].label

    return relPrint

# returns the route(that you get from navigateTo function) as a string with braces and
# arrow marks separating the spaces
def printRoute(route):
    rtPrint = ''
    l = len(route)

    if l == 1:
        return route[0][0].label
    
    i = 0
    while i < l-1:
        if route[i+1][1] == 0:
            rtPrint += route[i][0].label + ' -> '
        elif route[i+1][1] == 1:
            rtPrint += route[i][0].label + ' ( '
        elif route[i+1][1] == -1:
            rtPrint += route[i][0].label + ' ) '
            
        i += 1

    rtPrint += route[l-1][0].label

    return rtPrint

class connection:
    def __init__(self, source, destination):
        self.fromSpace = source
        self.toSpace = destination

        #print('new connection between '+source.label+' and '+destination.label)
        
        self.id = uuid.uuid4()
        if self.id in connectionList:
            reportError('Id clash for this connection')
        else:
            connectionList[self.id] = self