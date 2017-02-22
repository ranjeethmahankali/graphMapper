from spaceGraph import *

spaceList = ['red',
            'green',
            'blue',
            'yellow',
            'white']

#this is a modified implementation of the cSpace class that is customized for tensorflow
class space(cSpace):
    def __init__(name, childList, parentSpace = None):
        cSpace. __init__(name, parentSpace)
        # this is the list which has all the children
        # this is needed because we need indices, and the original children dictionary in
        # cSpace class does not provide any indices
        self.cL = childList
        # orientation property can either be 1 or 0, 0 implies landscape, 1 implies portrait 
        self.orientation = 0
        # this is the list of points that is used to calculate the wall positions
        self.ptList = list()
        # this is a list of points, one corresponding to one wall, the orientation of the wall is
        # determined by the orientation of the space that it is dividing
        self.wallPos = list()
        self.addChildren(childList)
    
    # this function will populate the ptList randomly
    # write this func and call it at the instance creation
    def populatePts():
        return

    # this function will calculate the positions of the walls
    # write this func and call it in a loop while creating the dataset
    def makewalls():
        return
        
    #returns the indices of the two spaces corresponding to the connection number
    def getConSpaces(num):
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
    def getConIndex(sA,sB):
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