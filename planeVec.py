#this is the 2d vector functions library
import math
print('testing')

#returns the distance between the two points
def dist(x1,y1,x2,y2):
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

#returns the inclination of the line with positive x-axis in radians
def lineAngle(x0,y0,x1,y1):
    angle = None

    if x1 > x0:
        if y1 > y0:
            angle = math.atan((y1-y0)/(x1-x0))
        elif y1 == y0:
            angle = 0;
        else:
            angle = math.atan((y1-y0)/(x1-x0)) + (2*math.pi)
    elif x1 == x0:
        if y1 > y0:
            angle = math.pi/2
        elif y1 == y0:
            angle = None
        else:
            angle = (3*math.pi)/2
    else:
        if y1 > y0:
            angle = math.atan((y1-y0)/(x1-x0)) + math.pi
        elif y1 == y0:
            angle = math.pi
        else:
            angle = math.atan((y1-y0)/(x1-x0)) + math.pi

    return angle

#returns the magnitude of a vector
def mod(vec):
    return math.sqrt((vec[0]**2)+(vec[1]**2))

#returns the dot product of these two vectors
def dot(vec1, vec2):
    return (vec1[0]*vec2[0])+(vec1[1]*vec2[1])

#returns the cos of the angle between these two vectors
def cosAng(vec1, vec2):
    if (mod(vec1) != 0 and mod(vec2) != 0):
        return dot(vec1,vec2)/(mod(vec1)*mod(vec2))
    else:
        return None
        print("Division by zero in cosAng function")

#returnss the product of the vector with scalar
def vPrd(vec, sc):
    return [vec[0]*sc, vec[1]*sc]

#returns the su of the two vectors
def vSum(vec1, vec2):
    return [vec1[0]+vec2[0], vec1[1]+vec2[1]]

#returns vec1-vec2
def vDiff(vec1, vec2):
    return [vec1[0]-vec2[0], vec1[1]-vec2[1]]

#returns the unit vector parallel to the given vector
def unitV(vec):
    length = mod(vec)
    if length == 0:
        return [0,0]
    else:
        return [vec[0]/length, vec[1]/length]

#gives a vector that is a perpendicular dropped from p to the line joining p1 and p2
def lineDist(p1, p2, p):
    line = vDiff(p2, p1) #line joining p1 to p2
    vec = vDiff(p, p1) #line joining p1 to p

    uLine = unitV(line) #unit vector along line vector
    length = dot(line, vec)/mod(line) #length of component of vec along line

    dLine = vPrd(uLine, length)

    return vDiff(dLine, vec)

#this method returns the cross product of the two vectors v1 and v2
#in the order v1 x v2
#since this is a plane vector library and all the vectors are 2D, the cross product
#is always perpendicular to the place and has no way to be represented in this library
#which doesn't supprt a third term in the ordered pair.
#hence This method will only return the magnitude of the cross product
def vCross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

#this method returns the closes point from the list pArr to pos
#this method is a replacement for what is found in the rhinoscriptsyntax module
def pointArrayClosestPoint(pArr, pos):
    minDist = math.inf
    closestPt = None

    i = 0
    while i < len(pArr):
        if mod(vDiff(pos,pArr[i])) < minDist:
            closestPt = pArr[i]
        i += 1

    return closestPt

#this method returns the intersection point of line joining a1 and a2
#and the line joining b1 and b2
def intersectionPt(a1, a2, b1, b2):
    #this method returns the point of intersection of two lines.
    #one is a segment from a1 to a2 and the other one is a segment from b1 to b2

    uA = unitV(vDiff(a2,a1))
    uB = unitV(vDiff(b2,b1))
    UAxUB = vCross(uA, uB)

    if UAxUB == 0:
        #the lines are parallel so there is no interesection
        return None
    else:
        aParam = (vCross(vDiff(b1,a1),uB))/UAxUB

        intPt = vSum(a1, vPrd(uA, aParam))
        #the above point is the intersection point but now we have to check
        #if it lies on both the segments

        checkA = dot(vDiff(intPt, a1), vDiff(intPt, a2))
        checkB = dot(vDiff(intPt, b1), vDiff(intPt, b2))

        #print(a1, a2, b1, b2)

        if checkA <= 0 and checkB <=0 :
            return intPt
        else:
            #this means the infinite lines intersect but the segments don't
            return None
