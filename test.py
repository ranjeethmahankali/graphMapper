def getConSpaces(numSpace, num):
    possibleConnections = numSpace*(numSpace-1)/2
    if num > possibleConnections - 1:
        print('The connection index exceeds the total possible connections')

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

print(getConSpaces(6,14))

def getConIndex(numSpace,sA,sB):
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

print(getConIndex(6,3,5))