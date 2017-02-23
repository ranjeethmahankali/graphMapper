from spaceGraph import *
from PIL import Image, ImageDraw
import random
import planeVec as pv

#this is a modified implementation of the cSpace class that is customized for tensorflow
class space(cSpace):
    def __init__(self, name, childList, dimensions, parentSpace = None):
        cSpace.__init__(self,name, parentSpace)
        # this is the list which has all the children
        # this is needed because we need indices, and the original children dictionary in
        # cSpace class does not provide any indices
        self.cL = childList
        # this list has the dimensions of the space in [width, height] format
        self.dim = dimensions
        # orientation property can either be 1 or 0, 0 implies landscape, 1 implies portrait 
        self.orientation = self.getOrientation(self.dim)
        # this is the list of points that is used to calculate the wall positions
        self.ptList = list()
        # this is a list of items, with each being a list having an end point start and end point of the wall
        self.walls = list()
        # adding a new space as a child for every name in the childList
        self.addChildren([cSpace(cName) for cName in self.cL])
    
    # this function will return the value for the orientation depending on the dimensions - dims param
    def getOrientation(self,dims):
        if dims[0] >= dims[1]:
            return 0
        else:
            return 1

    # this function will populate the ptList randomly
    # write this func and call it at the instance creation
    def populatePts(self):
        self.ptList = []
        for i in range(len(self.cL)):
            xPos = random.randint(1, self.dim[0])
            yPos = random.randint(1, self.dim[1])
            self.ptList.append([xPos, yPos])
            
        return

    # this function will calculate the positions of the walls
    # write this func and call it in a loop while creating the dataset
    def makeWalls(self, split = None, new=True):
        # split is a list of jobs to do. Each job is a set of points and the x,y limits of the space
        # they occupy. The job needs to be popped from the list, then points need to be split with a
        # new wall, resulting in two new jobs. This recursion will continue unless the job has only one
        # point in it.
        
        #default values
        if split is None:
            split = [{'pt':self.ptList,'x0':0,'y0':0,'x1':self.dim[0],'y1':self.dim[1]}]
        
        #resetting walls if new
        if new:
            self.walls = []

        job = split.pop()
        if len(job['pt']) <= 1:
            return
        
        wallPt = pv.meanVec(job['pt'])
        dims = [job['x1'] - job['x0'], job['y1'] - job['y0']]
        orn = self.getOrientation(dims)
        startPt = None
        endPt = None
        job1 = None
        job2 = None
        if orn == 0:
            #do sth for landscape
            startPt = [wallPt[0], job['y0']]
            endPt = [wallPt[0], job['y1']]
            #split the job into two
            job1 = {'pt':list(), 'x0': job['x0'], 'x1':wallPt[0], 'y0':job['y0'], 'y1':job['y1']}
            job2 = {'pt':list(), 'x0':wallPt[0], 'x1': job['x1'], 'y0':job['y0'], 'y1':job['y1']}
            # now distribute points between the new jobs
            for pt in job['pt']:
                if pt[0] <= wallPt[0]:
                    job1['pt'].append(pt)
                else:
                    job2['pt'].append(pt)
        elif orn == 1:
            # do sth for portrait
            startPt = [job['x0'], wallPt[1]]
            endPt = [job['x1'], wallPt[1]]
            #split the job into two
            job1 = {'pt':list(), 'x0': job['x0'], 'x1': job['x1'], 'y0':job['y0'], 'y1':wallPt[1]}
            job2 = {'pt':list(), 'x0': job['x0'], 'x1': job['x1'], 'y0':wallPt[1], 'y1':job['y1']}
            # now distribute points between the new jobs
            for pt in job['pt']:
                if pt[1] <= wallPt[1]:
                    job1['pt'].append(pt)
                else:
                    job2['pt'].append(pt)
                
        self.walls.append([startPt, endPt])
        split += [job1, job2]
        self.makeWalls(split, new = False)
        
        return
        
    #returns the indices of the two spaces corresponding to the connection number
    def getConSpaces(self, num):
        numSpace = len(self.cL)
        possibleConnections = numSpace*(numSpace-1)/2
        if num > possibleConnections - 1:
            print('The connection index exceeds the total possible connections')
            return None

        s1 = None
        s2 = None
        i = 1
        l = 0 #lower limit
        t2 = 0
        while i < numSpace:
            # calculating the upper limit
            t2 = t2+i
            u = i*numSpace - t2
            if l <= num < u:
                s1 = i-1
                s2 = i + (num-l)
                break
            
            l = u
            i += 1

        return [s1,s2]

    #returns the index of te connection between the two spaces, whose indices are supplied as parameters.
    def getConIndex(self, sA,sB):
        numSpace = len(self.cL)
        if sA >= numSpace or sB >= numSpace:
            print('Invalid space indices')
            return None

        s1 = min(sA, sB)
        s2 = max(sA, sB)

        l = 0
        step = numSpace - 1
        i = 0
        while i < s1:
            l += step
            step -= 1
            i += 1
        
        ans = l + s2 - s1 -1
        return ans

# the main logic begins here
# this is the space list for the dataset
spaceList = ['red',
            'green',
            'blue',
            'yellow',
            'white']

sample = space('sample', spaceList, [64,48])

# for i in range(dataNum):
    # populate sample with pts
    # make walls
    # export the training example
sample.populatePts()
sample.makeWalls()