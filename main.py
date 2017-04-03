import math, time, cv2
import numpy as np
from hand import *
from coords import *
from depthmap import *
from contours import *
from handstats import *
from circles import *
from skin import SkinDetector

a= []

def main():
    hand = HandStats(isRight=True)
    skin_detector = SkinDetector()
    while True:
        depthMap = getDepthMap()
        filt_depthMap = cv2.GaussianBlur(depthMap, (11,11), 0)
        cv2.medianBlur(filt_depthMap, 11, dst=filt_depthMap)
        mask = getMask(filt_depthMap)

        struc = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
        cv2.erode(mask, struc)
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, struc)
        cv2.dilate(mask, struc)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struc)

        # cv2.GaussianBlur(hsvmask, (3, 3), 0, hsvmask)
        
        # cv2.imshow('test4',mask)
        # # cv2.imshow('test2',depthMap)
        # # cv2.imshow('test3',filt_depthMap)
        # k = chr(cv2.waitKey(10)& 0xff)
        # if k == 'c':
        #     break
        # else:
        #     continue

        #cv2.drawContours(depthMap,contours,-1,(0,0,255),3)
        img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if hand.isCalibrated() and hand.isOnScreen(mask):
            #print hand.isOnScreen(mask)
            img = drawHand(img, hand, mask)
            #print hand.palmCirc.getRadius()
        else:
            contours = getContours(mask)
            if len(contours)==0: 
                continue
            #print hand.isOnScreen(mask)
            biggestCnt = getBiggestContour(contours)
            img = highlightCnt(img, biggestCnt)
            img = drawTextInCorner(img, 'Hand Not Found')
        cv2.imshow("test",img)
        if handleKeyResponse(img, hand, mask) == True:
            continue
        else:
            break


def drawHand(img, hand, mask):
    fingDict = hand.isOnScreen(mask)
    if fingDict != False:
        # handCnt = hand.findHandCnt(mask)
        # defectPnts = getContourConvexDefects(handCnt, minSize=15, maxSize=80)
        handCirc, handCnt, defectPnts,highestPnt= hand.getPalmCircle(mask)
        # cv2.circle(img, highest_defectPnt.toTuple(), 10, (87,127,35), 2)
        img = highlightCnt(img, handCnt)
        # poly = getApproxContourPolygon(handCnt, accuracy=0.02)
        # for pnt in poly:
        #     cv2.circle(img, pnt.toTuple(), 10, (127,127,127), 2)
        # img = drawCntPolygon(img, handCnt,handCirc)
        # img = drawFingPoints(img, hand, mask,openFingLs=fingDict)
        img = drawPalmCircle(img,handCirc)
        img = drawTextInCorner(img, 'Hand Found')
    return img


def drawFingPoints(img, hand, mask,openFingLs=None):
    handCnt = hand.findHandCnt(mask)
    defectPnts = getContourConvexDefects(handCnt, minSize=15, maxSize=80)
    if len(defectPnts) > 0:
            highestPnt = min(defectPnts, key=lambda pnt: pnt.getY())
    else:
            highestPnt = Point(480,480)
    fingPnts = getOpenFingerPnts(mask,hand.findHandCnt(mask),highestPnt, palmCirc=hand.getPalmCircle(mask)[0])
    # print len(fingPnts)
    for pnt in fingPnts: cv2.circle(img, pnt.toTuple(), 10, (127,127,127), 2)
    return img


def drawCntPolygon(img, cnt,handCirc):
    poly = getApproxContourPolygon(cnt, accuracy=0.005)
    for i in range(len(poly)-1):
        cv2.line(img, poly[i].toTuple(), handCirc.getCenter().toTuple(), (0, 0, 255), 3)
    return img


def highlightCnt(img, cnt):
    cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    return img


def drawPalmCircle(img, handCirc):
    cv2.circle(img, handCirc.getCenter().toTuple(), int(handCirc.getRadius()), (255, 0, 0), 3)
    cv2.circle(img, handCirc.getCenter().toTuple(), 5, (255, 0, 0), -1)
    return img


def drawTextInCorner(img, text):
    cv2.putText(img, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 8)
    return img


def handleKeyResponse(img, hand, mask):
    key = chr(cv2.waitKey(10) & 0xFF)  # if 64 bit system, waitKey() gives result > 8 bits, ANDing with 11111111 removes extra ones
    if key == 'c':   
        hand.calibrate(mask)
        a.append(1)
    elif hand.isOnScreen(mask):
        if key == 'g': 
            openFingers = hand.getOpenFingers(mask)
            if openFingers == None:
                pass
            else:
                print openFingers
        elif key == 'v':
            handPosList = hand.getHandStaticGesture(sampleTimeMsec=150, sampIntervalMsec=10)
            print "begin: "
            print handPosList
            print "end"
            # print velocVec.toTuple() if velocVec != None else "no result"
        elif key == 'a':
            accVec = hand.getHandAccelVec(sampleTimeMsec=200, sampIntervalMsec=10)
            print accVec.toTuple() if accVec != None else "no result"
        elif key == 's':
            return False
    if len(a) != 0:
        clock_begin = time.time()
        while(time.time() - clock_begin)<50/1000:
            pass
        hand.getHandStaticGesture(sampleTimeMsec=150, sampIntervalMsec=10)
    return True



if __name__ == "__main__":
    main()
