import sys
import spaceGraph as sg
from ops import *
from PIL import Image, ImageDraw
import random
import planeVec as pv
import math

default_coords = {'pt':[], 'x0':0,'x1':imgSize[0],'y0':0,'y1':imgSize[1]}

# this is the 'wall' object
class wall:
    def __init__(self, startPt, endPt, drawColor = '#000000', owner = None):
        self.start = startPt
        self.end = endPt
        self.color = drawColor
        # this is the list of all doors located on this wall
        self.doors = list()
        self.owner = owner #this is the owner space of this wall
        self.owner.walls.append(self)
        # this is the list of neighbors, should be of length no more than 2
        self.nbrs = list()
    
    # this returns the length of the wall
    def length(self):
        return pv.mod(pv.vDiff(self.start, self.end))
    # this method draws the wall onto the provided PIl img - pending
    def render(self, img = Image.new("RGB", spaceSize, "white"), color=0):
        draw = ImageDraw.Draw(img)
        draw.line(self.start + self.end, fill = color)
        del draw
        return img
    # this wall updates the neighbors
    def updateNeighbors(self):
        self.nbrs = list()
        midPt = pv.vPrd(pv.vSum(self.start, self.end), 0.5)
        ln = pv.unitV(pv.vDiff(self.end, self.start))
        step = pv.vRotate(ln, math.pi/2)
        pt1 = pv.vSum(midPt, step)
        pt2 = pv.vDiff(midPt, step)

        test = False
        for label in self.owner.c:
            if self.owner.c[label].hasPoint(pt1) or self.owner.c[label].hasPoint(pt2):
                self.nbrs.append(self.owner.c[label])
                test = True
        
        # print('%s neighbors'%len(self.nbrs))
        
    # this method removes all doors
    def removeAllDoors(self):
        self.doors = list()
    # this is the default string parsing
    def __str__(self):
        return "[%s, %s]"%(self.start, self.end)


# this is the door class
class door:
    def __init__(self, parentWall, doorPos, doorSize = 1, doorColor=(255,255,255)):
        self.size = abs(doorSize)
        self.wall = parentWall
        if self.wall.length() < self.size+3:# exiting because the wall is too small
            del self
            return
        parentWall.doors.append(self)
        # this is the parameter between 0 and 1 representing the position of door on the line
        self.pos = doorPos
        # this color will be used to render the door
        self.color = doorColor
        
        # print(len(self.wall.nbrs))
        self.wall.owner.connectChildren(self.wall.nbrs[0].label, self.wall.nbrs[1].label)
    
    def render(self, img):
        draw = ImageDraw.Draw(img)
        # offset the door from ends of wall to avoid awkward joints
        offset = math.ceil((self.size / 2)+1)
        offsetRatio = offset / (pv.mod(pv.vDiff(self.wall.end, self.wall.start)))
        # these are the start and end points of the wall after accounting for the offset
        newStart = pv.linEval(self.wall.start, self.wall.end, offsetRatio)
        newEnd = pv.linEval(self.wall.end, self.wall.start, offsetRatio)
        # now evaluating door position between these new start and new end points
        xypos = pv.linEval(newStart, newEnd, self.pos)
        # draw the door
        hSize = self.size/2
        coord = [xypos[0] - hSize,
                xypos[1] - hSize,
                xypos[0] + hSize,
                xypos[1] + hSize]
        
        draw.rectangle(coord, self.color)

        del draw
        return img
    
    # default string parsing
    def __str__(self):
        return "Door at %s"%(self.pos)

# this is a modified implementation of the cSpace class that is customized for tensorflow
class space(sg.cSpace):
    def __init__(self, name, childNames, coord={'pt':[],'x0':0,'y0':0,'x1':spaceSize[0],'y1':spaceSize[1]}, parentSpace = None):
        sg.cSpace.__init__(self, name, parentSpace)
        # this is the list which has all the children
        # this is needed because we need indices, and the original children dictionary in
        # cSpace class does not provide any indices
        # this list has the dimensions of the space in [width, height] format
        self.coord = coord
        self.dim = [self.coord['x1'] - self.coord['x0'], self.coord['y1'] - self.coord['y0']]
        # orientation property can either be 1 or 0, 0 implies landscape, 1 implies portrait 
        self.orientation = self.getOrientation(self.dim)
        # this is the list of points that is used to calculate the wall positions
        self.ptList = coord['pt']
        # this is a list of items, with each being a list having an end point start and end point of the wall
        self.walls = list()
        # shuffling the names just before making them a property so that they come out different
        # everytime, more diversity in the dataset
        self.childNames = childNames[:]
        random.shuffle(self.childNames)
        # print(self.childNames)
    
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
        for i in range(len(self.childNames)):
            xPos = random.randint(1, self.dim[0])
            yPos = random.randint(1, self.dim[1])
            self.ptList.append([xPos, yPos])
            
        return

    # this function returns if this space has a point inside it or not
    def hasPoint(self, pt):
        if (self.coord['x0']< pt[0] <self.coord['x1']) or (self.coord['x1']< pt[0] <self.coord['x0']):
            if (self.coord['y0']< pt[1] <self.coord['y1']) or (self.coord['y1']< pt[1] <self.coord['y0']):
                return True
        
        return False
    # this function will calculate the positions of the walls and adds them to the walls list
    def makeWalls(self, split = None, new=True):
        # split is a list of jobs to do. Each job is a set of points and the x,y limits of the space
        # they occupy. The job needs to be popped from the list, then points need to be split with a
        # new wall, resulting in two new jobs. This recursion will continue unless the job has only one
        # point in it.

        #default values
        if split is None:
            split = [{'pt':self.ptList,'x0':0,'y0':0,'x1':self.dim[0],'y1':self.dim[1]}]
        
        if len(split) == 0:#nothing to do
            return

        #resetting walls if new
        if new:
            self.walls = []

        job = split.pop()
        if len(job['pt']) == 0:
            return
        if len(job['pt']) == 1:#nothing to divide
            #create space and add as a child to this space
            # print('here2')
            newSpace = space(self.childNames.pop(),[], job, self)
        else:
            wallPt = pv.meanVec(job['pt'])
            dims = [job['x1'] - job['x0'], job['y1'] - job['y0']]
            orn = self.getOrientation(dims)
            startPt = None
            endPt = None
            job1 = None
            job2 = None
            if orn == 0:
                # print('x')
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
                # print('y')
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
            
            # this constructor automatically adds the wall to the self.walls list
            newWall = wall(startPt, endPt, owner = self)
            
            if len(job1['pt']) == 2 and len(job2['pt']) == 0:
                job2['pt'].append(job1['pt'].pop())
            elif len(job2['pt']) == 2 and len(job1['pt']) == 0:
                job1['pt'].append(job2['pt'].pop())

            split += [job1, job2]
            # print(len(job1['pt']), len(job2['pt']))
            # print(job1['pt'], job2['pt'])

        self.makeWalls(split, new = False)
        return
        
    # this method splits the created walls at every intersection and also updates neighbor info
    def splitWalls(self):
        newWallList = list()
        # this nested def splits the wall w1 at the intersection point intPt
        def splitWall():
            if intPt is None:
                return 0
            # print(w1, w2, intPt)
            dist = min(pv.mod(pv.vDiff(w1.start, intPt)), pv.mod(pv.vDiff(w1.end, intPt)))
            if dist < 0.1:
                return 0
            
            split1 = wall(w1.start, intPt, owner = self)
            split2 = wall(intPt, w1.end, owner = self)
            # print('added 2 walls')
            return 1

        # getting back to the original definition
        while len(self.walls) > 0:
            # print('walls: %s %s'%(len(self.walls), len(newWallList)))
            w1 = self.walls.pop()
            # print('walls: %s %s'%(len(self.walls), len(newWallList)))
            splitFirst = -1
            splitSecond = -1
            for w2 in self.walls:
                intPt = pv.intersectionPt(w1.start,w1.end,w2.start,w2.end)
                splitFirst = splitWall()
                if splitFirst == 1:
                    break

            if not splitFirst == 1:
                for w2 in newWallList:
                    intPt = pv.intersectionPt(w1.start,w1.end,w2.start,w2.end)
                    splitSecond = splitWall()
                    if splitSecond == 1:
                        break
            
            # print("splits: %s %s"%(splitFirst, splitSecond))
            isUnbreakable= (splitFirst == 0 and splitSecond == 0)or(splitFirst == 0 and splitSecond == -1)
            isUnbreakable = isUnbreakable or (splitFirst == -1 and splitSecond == 0)
            if isUnbreakable:
                newWallList.append(w1)

        # finally updating the walls with new walls
        # print(len(newWallList))
        self.walls = newWallList
        # updaitng neighbor information
        for w in self.walls:
            w.updateNeighbors()
                    
    # this randomly populates the walls with doors
    def makeRandDoors(self):
        for w in self.walls:
            # removing any doors that were previously created
            if len(w.nbrs) < 2:
                continue
            # this is the range to make random decision
            dec = [0.3, 0.5]
            if random.uniform(0,1) > random.uniform(dec[0], dec[1]):
                d = door(w, random.uniform(0,1))
    # this removes all doors and connections between child spaces
    def removeDoors(self):
        for w in self.walls:
            w.removeAllDoors()
        
        for cName in self.c:
            self.c[cName].con = dict()
            self.c[cName].connected = set()

    # this returns the flat version of the graph of it's children's connection
    def getFlatGraph(self):
        num = len(nameList)
        maxCon = int(num*(num-1)*0.5)
        flatGraph = [0]*maxCon
        # print(len(flatGraph))
        for i in range(len(flatGraph)):
            j, k = self.getConSpaces(i)
            if self.c[nameList[j]].isConnected(self.c[nameList[k]]):
                flatGraph[i] = 1
        
        return flatGraph

    # this def renders the space into an image using PIL and returns that img
    def render(self, showPt = False):#showPt param decides whether to show pts or not - pending
        global colors
        background = Image.new("RGB", self.dim, "white")
        wallLayout = Image.new("RGBA", self.dim)
        doorMask = Image.new("RGBA", self.dim)
        mask = Image.new("RGBA", self.dim)
        # draw room background colors on background
        # with ImageDraw.Draw(background) as draw:
        draw = ImageDraw.Draw(background)
        for cName in self.c:
            box = [self.c[cName].coord['x0'], 
                    self.c[cName].coord['y0'],
                    self.c[cName].coord['x1'], 
                    self.c[cName].coord['y1']]
            draw.rectangle(box,colors[cName])
        del draw

        # draw walls on the wallLayout
        for wall in self.walls:
            wallLayout = wall.render(wallLayout)
            mask = wall.render(mask, color=(255,255,255))
            # render doors here
            for dr in wall.doors:
                doorMask = dr.render(doorMask)
        # create openings in the wall layout
        # draw wallLayout image on top of background
        finalPlan = Image.new("RGB", self.dim, "white")
        finalPlan.paste(background, (0,0))
        finalPlan.paste(wallLayout.convert("RGB"), (0,0), mask)
        finalPlan.paste(background, (0,0), doorMask)
        # mask.save('mask2.png')
        # convert the whole thing into RGB and return it
        return finalPlan
    
    # This def calculates the wall that separates two particular spaces and returns the index
    def borderWall(): #- pending
        return
    # #returns the indices of the two spaces corresponding to the connection number
    def getConSpaces(self, num):
        numSpace = len(nameList)
        possibleConnections = int(numSpace*(numSpace-1)/2)
        if num > possibleConnections - 1:
            print('flatgraph index too high. %s > %s'%(num, possibleConnections))
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
        numSpace = len(nameList)
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