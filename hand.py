import cv2, math, time
from coords import *
from depthmap import *
from circles import *
from contours import *
import numpy as np



# def getOpenFingerVectors(test,handCnt, defectPnts,palmCirc=None, isRightHand=True):
#     if palmCirc==None or handCnt==None: return None
#     cent = palmCirc.getCenter()
#     fingOffsetsFromMid = getFingIndexOffsetsFromMidFing(isRightHand=isRightHand)
#     fingPnts = getOpenFingerPnts(test,handCnt,defectPnts, palmCirc=palmCirc)
#     midFingIndex = getMidFingIndex(fingPnts)
#     fingNameLs = getFingList(isRightHand=isRightHand)
#     fingPntDict = {fing: fingPnts[(midFingIndex + fingOffsetsFromMid[fing]) % len(fingPnts)] for fing in fingNameLs}
#     return {fing: cent.getVectorTo(fingPntDict[fing]) for fing in fingNameLs}


def getOpenFingerPnts(mask, handCnt, defectPnts,highestPnt, palmCirc=None):
    if palmCirc==None or handCnt==None: return None
    fingPnts = []
    poly = getApproxContourPolygon(handCnt, accuracy=0.008)
    cent = palmCirc.getCenter(); rad = palmCirc.getRadius()

    # highestPnt = min(getCntPntLs(handCnt), key=lambda pnt: pnt.getY())
    highest_cent = highestPnt.getDistTo(palmCirc.getCenter())
    for i in range(len(poly)-1):
        # fingers at accute angle points, accute if adjacent points are closer to center than it
        pnt = poly[i]; pntAfter = poly[(i+1)%len(poly)]; pntBefore = poly[(i-1)%len(poly)]
        if highest_cent > 110:
            if pntAfter.getDistTo(cent) <= pnt.getDistTo(cent) and pntBefore.getDistTo(cent) <= pnt.getDistTo(cent):
                if pnt.getDistTo(cent) > (rad+40) and (pnt.getY() <= cent.getY() or abs(pnt.getY()-cent.getY()) < 10):
                    fingPnts.append(pnt)
        else:
            if pntAfter.getDistTo(cent) <= pnt.getDistTo(cent) and pntBefore.getDistTo(cent) <= pnt.getDistTo(cent):
                if pnt.getDistTo(cent) > (rad+17) and (pnt.getY() <= cent.getY() or abs(pnt.getY()-cent.getY()) < 10):
                    fingPnts.append(pnt)

                # print pnt.getDistTo(cent)-rad-46
    # print len(fingPnts)
    # if len(fingPnts) == 5:       
    #     print "right"
    # elif len(fingPnts)  != 2 and len(fingPnts) != 3:
    #     for i in range(len(poly)-1):
    #         cv2.imshow('find',test)
    #         cv2.line(test, poly[i].toTuple(), cent.toTuple(), (36, 127, 21), 3)
    #         cv2.waitKey(0) 
    #         print poly[i].getDistTo(cent)-rad-41
    #         print -poly[i].getY()+cent.getY()
    #         pnt = poly[i]; pntAfter = poly[(i+1)%len(poly)]; pntBefore = poly[(i-1)%len(poly)]
    #         print pnt.getDistTo(cent)-pntAfter.getDistTo(cent)
    #         print pnt.getDistTo(cent)-pntBefore.getDistTo(cent)
    #         print highest_cent
    # else:
    #     print "wrong"
    # print len(fingPnts)
    return fingPnts


# def getFingAngRegions(test,handCnt, defectPnts,palmCirc, isRightHand=True):
#     fingNames = getFingList(isRightHand=isRightHand)
#     fingVecs = getOpenFingerVectors(test,handCnt, defectPnts,palmCirc, isRightHand=isRightHand)
#     betweenFingAngs = getAngsBetweenVecs([fingVecs[fing] for fing in fingNames])
#     fingAngBounds = list(reversed(sorted([math.pi]+betweenFingAngs+[0])))  # angs from pi to 0, all angles between finger vectors
#     # fingNames in order from leftmost finger's name to right
#     # leftmost finger has largest angle to horizontal (<1,0>), so its boundaries = 1rst items in fingAngBounds
#     return {fingNames[i]: sorted([fingAngBounds[i], fingAngBounds[i+1]]) for i in range(len(fingNames))}


def getFingIndexOffsetsFromMidFing(isRightHand=True):
    fingers = getFingList(isRightHand=isRightHand)
    fingOffsetsFromMiddle = {fing: fingers.index(fing) - fingers.index('middle') for fing in fingers}
    return fingOffsetsFromMiddle


def getMidFingIndex(hullPnts):
    midFingCoords = min(hullPnts, key=lambda pnt: pnt.getY())  # assume highest point on hull = middle finger
    return hullPnts.index(midFingCoords)


def getAngsBetweenVecs(vecLs):
    vecAngLs = [vec.getAngFromHoriz() for vec in vecLs]
    return [average([vecAngLs[i], vecAngLs[i+1]]) for i in range(len(vecAngLs)-1)]


def getFingList(isRightHand=True):
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    return list(reversed(fingers)) if isRightHand else fingers


def getHighestNotFingPnt(hand, mask):
    handCntPnts = getCntPntLs(hand.findHandCnt(mask))
    noFingerRange = Circle(hand.palmCirc.getCenter(), hand.palmCirc.getRadius()+20)  # make sure highest point for palm isn't fingertip
    cntPntsInErrorBounds = filter(lambda pnt: noFingerRange.containsPnt(pnt), handCntPnts)
    if len(cntPntsInErrorBounds)>0: highestNotFingPnt = min(cntPntsInErrorBounds, key=lambda pnt: pnt.getY())
    else:                           highestNotFingPnt = min(handCntPnts, key=lambda pnt: pnt.getY())
    return highestNotFingPnt



class Hand(object):
    def __init__(self, isRight=True):
        self.fingAngRegions = {}
        self.palmCirc = None
        self.handArea = 0
        self.calibrated = False
        self.isRight = isRight

    def calibrate(self,mask):
        handCnt = getBiggestContour(getContours(mask))
        self.palmCirc,handCnt,defectPnts,highestPnt = self.getPalmCircle(mask)
        # self.fingAngRegions = getFingAngRegions(mask,handCnt, defectPnts, self.palmCirc, isRightHand=self.isRight)
        self.handArea = cv2.moments(handCnt)['m00']
        self.calibrated = True

    def findHandCnt(self, mask):
        contours = getContours(mask)
        # will be None if no viable hand contour found
        if not self.calibrated: return getBiggestContour(contours)
        else: return getContourWithArea(contours, self.handArea, floor=self.handArea/5, ceil=self.handArea*2.5)

    def getOpenFingers(self, mask):
        if not self.isOnScreen(mask): return None
        # handCnt = self.findHandCnt(mask)
        # defectPnts = getContourConvexDefects(handCnt, minSize=15, maxSize=80)
        if len(defectPnts) > 0:
            highestPnt = min(defectPnts, key=lambda pnt: pnt.getY())
        else:
            highestPnt = Point(480,480)
        Circle = self.getPalmCircle(mask)
        if Circle == None:
            return None
        circle = Circle[0]
        handCnt = Circle[1]
        defectPnts = Circle[2]
        fingVecs = [circle.getCenter().getVectorTo(pnt) for pnt in getOpenFingerPnts(mask,handCnt,highestPnt, circle)]
        openFingers = {}
        # for finger in getFingList(isRightHand=self.isRight):
        #     openFingers[finger] = any([self.fingAngRegions[finger][0] <= vec.getAngFromHoriz() <= self.fingAngRegions[finger][1] for vec in fingVecs])
        return openFingers

    def getHandPos(self, mask):
        if not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        self.palmCirc = self.getPalmCircle(mask)[0]
        return self.palmCirc.getCenter()

    def getPalmCircle(self, mask):
        """Assume that hand is open on first run."""
        if self.calibrated and not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        defectPnts = getContourConvexDefects(handCnt, minSize=15, maxSize=80)
        palmCircPnts = []
        palmCircPnts += defectPnts
        Rdius = 0

        outp = cv2.distanceTransform(mask, cv2.cv.CV_DIST_L2, 5)
        highestPnt = min(getCntPntLs(handCnt), key=lambda pnt: pnt.getY())
        # if len(defectPnts) > 0:
        #     defecthighestPnt = min(defectPnts, key=lambda pnt: pnt.getY())
        # else:
        #     defecthighestPnt = Point(480,480)

        i = np.argsort(-outp,axis=None)
        for i_index in range ((len(i)-1)/50):
            max_index = np.unravel_index(i[50*i_index], outp.shape)
            max_index = max_index[1], max_index[0]
            # max_val = outp[max_index[1]][max_index[0]]
            palm_pos = np.array(max_index)
            if Point(palm_pos[0],palm_pos[1]).getDistTo(highestPnt) <= 140:
                break
        #     palmCircPnts.append(getHighestNotFingPnt(self, mask))

        # NearestPnt = Point(100,100)

        if self.calibrated and self.isOnScreen(mask):
            Rdius = self.palmCirc.getRadius()
        elif self.palmCirc!=None and (defectPnts==None or len(defectPnts)!=6):
            Rdius = 38
        else: 
            palmCircPnts += defectPnts
            NearestPnt = min(palmCircPnts, key=lambda pnt: pnt.getDistTo(Point(palm_pos[0],palm_pos[1])))
            Rdius = NearestPnt.getDistTo(Point(palm_pos[0],palm_pos[1]))


        # print highestPnt.getDistTo(Point(palm_pos[0],palm_pos[1]))

        # if self.calibrated == False:
        #     Rdius = 38
        # print  defecthighestPnt.getDistTo(highestPnt)
        #print Rdius
        #lowestPnt = max(getCntPntLs(handCnt), key=lambda pnt: pnt.getY())
        #palmCircPnts.append(lowestPnt)
        #return getSmallestEnclosingCirc(palmCircPnts)
        # print Point(palm_pos[0],palm_pos[1]).getDistTo(defecthighestPnt)
        #print len(defectPnts)
        # distMap = np.zeros(mask.shape, dtype=np.dtype('uint8'))
        # for cont in contours:
        #     if cv2.contourArea(cont) < 1000:
        #         continue
        # depth8u = np.vectorize(lambda x: 255 if x > 0 else 0, otypes=['uint8'])(mask)
        # cv2.drawContours(depth8u, [cont, ], -1, [255], cv2.FILLED)
        # outp = cv2.distanceTransform(depth8u, cv2.cv.CV_DIST_L2, cv2.DIST_MASK_5)

        # print Rdius
        return Circle(Point(palm_pos[0],palm_pos[1]),Rdius),handCnt, defectPnts,highestPnt


    def isCalibrated(self):
        return self.calibrated

    def isOnScreen(self, mask):
        handCnt = self.findHandCnt(mask)
        return True if handCnt != None else False

    def getFinger_Sum(self,mask):
        # handCnt = self.findHandCnt(mask)
        # defectPnts = getContourConvexDefects(handCnt, minSize=15, maxSize=80)
        Circle = self.getPalmCircle(mask)
        if Circle == None:
            return None, None, None
        circle = Circle[0]
        handCnt = Circle[1]
        defectPnts = Circle[2]
        highestPnt = Circle[3]
        fingPnts = getOpenFingerPnts(mask,handCnt,defectPnts,highestPnt, circle)
        fingAngles = 0
        fingLength = 0
        if len(fingPnts) == 3:
            leftmost_Pnt = min(fingPnts, key=lambda pnt: pnt.getX())
            rightmost_Pnt = max(fingPnts, key=lambda pnt: pnt.getX())
            fingAngles = (leftmost_Pnt.getVectorTo(circle.getCenter())).getAngFromHoriz()-(rightmost_Pnt.getVectorTo(circle.getCenter())).getAngFromHoriz()
        elif len(fingPnts) == 1:
            highest = min(fingPnts, key=lambda pnt: pnt.getY())
            if len(defectPnts) == 0:
                fingLength = 0
            else:
                highest_defect = min(defectPnts, key=lambda pnt: pnt.getY())
                fingLength = highest.getDistTo(highest_defect)
        return len(fingPnts),fingAngles, fingLength

