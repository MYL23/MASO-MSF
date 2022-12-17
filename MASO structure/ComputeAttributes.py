# step 4: compute attributes based on GPS points in each pixel

import os
import numpy as np
import pickle
from geopy.distance import vincenty
import math
import copy
import operator
from functools import reduce
import gc
from tkinter import _flatten

Wp = Hp = 0.2  # spatial range
Wm = Hm = 32  # image size

# Identify the Speed  limit
SpeedLimit = {0: 7, 1: 12, 2: 120. / 3.6, 3: 180. / 3.6, 4: 120 / 3.6}


def calSpeed(pts, label):
    RDt, DTt, Speedt = [], [], []
    outlier = []
    for m in range(len(pts) - 1):
        A = (pts[m][0], pts[m][1])
        B = (pts[m + 1][0], pts[m + 1][1])
        RDt.append(vincenty(A, B).meters)
        DTt.append((pts[m + 1][2] - pts[m][2]) * 24. * 3600 + 1)
        tempSpeed = RDt[m] / DTt[m]
        if tempSpeed < 0 or tempSpeed > SpeedLimit[label]:
            outlier.append(m)
        Speedt.append(tempSpeed)
    Speedt = [q for p, q in enumerate(Speedt) if p not in outlier]
    return Speedt


def calAttr(pts, label):
    RDt, DTt, Speedt, Turnt = [], [], [], []
    outlier = []
    absDeltaSt = []
    absDelTurnt = []
    for m in range(len(pts) - 1):
        A = (pts[m][0], pts[m][1])
        B = (pts[m + 1][0], pts[m + 1][1])

        RDt.append(vincenty(A, B).meters)
        DTt.append((pts[m + 1][2] - pts[m][2]) * 24. * 3600 + 1)
        tempSpeed = RDt[m] / DTt[m]
        if tempSpeed < 0 or tempSpeed > SpeedLimit[label]:
            outlier.append(m)
        Speedt.append(tempSpeed)

        a = (pts[m + 1][0], pts[m + 1][1])
        b = (pts[m][0], pts[m + 1][1])
        c = (pts[m + 1][0], pts[m][1])
        turnRadians = math.atan2(vincenty(a, b).meters, vincenty(a, c).meters)  #
        turnDegree = (math.degrees(turnRadians) + 360) % 360
        Turnt.append(turnDegree)

    Speedt = [q for p, q in enumerate(Speedt) if p not in outlier]
    Turnt = [q for p, q in enumerate(Turnt) if p not in outlier]
    for p in range(len(Speedt) - 1):
        absDeltaSt.append(abs(Speedt[p + 1] - Speedt[p]))

    for p in range(len(Turnt) - 1):
        absDelTurnt.append(abs(Turnt[p + 1] - Turnt[p]))
    return Speedt, absDeltaSt, absDelTurnt


def statistics(data):
    tempCon = data[0][4]
    statis = 1
    for q in range(len(data)):
        if data[q][4] != tempCon:
            statis += 1
            tempCon = data[q][4]
    return statis


def splitPixelsPts(pts):
    splitlen = statistics(pts)
    splitPts = [[] for p in range(splitlen)]
    mark = pts[0][4]
    splitIndex = 0
    for m in range(len(pts)):
        if mark == pts[m][4]:
            splitPts[splitIndex].append(pts[m])
        else:
            splitIndex += 1
            mark = pts[m][4]
            splitPts[splitIndex].append(pts[m])
    return splitPts


def fillPts(filling, ptsData, num):
    oneSplit = copy.deepcopy(filling)
    if oneSplit[-1][0:4] != ptsData[oneSplit[-1][5]].tolist():
        print("********************* mismatch ***************")
    con = oneSplit[-1][4]
    if num == 1:
        if oneSplit[-1][5] + num > len(ptsData) - 1:
            index = oneSplit[0][5] - 1
            oneSplit.insert(0, [ptsData[index][0], ptsData[index][1], ptsData[index][2], ptsData[index][3], con,
                                -1])
        else:
            index = oneSplit[-1][5] + 1
            oneSplit.append(
                [ptsData[index][0], ptsData[index][1], ptsData[index][2], ptsData[index][3], con, -1])
    elif num == 2:
        if oneSplit[-1][5] == len(ptsData) - 1:
            index = oneSplit[-1][5]
            oneSplit.insert(0,
                            [ptsData[index - 1][0], ptsData[index - 1][1], ptsData[index - 1][2], ptsData[index - 1][3],
                             con,
                             -1])
            oneSplit.insert(0,
                            [ptsData[index - 2][0], ptsData[index - 2][1], ptsData[index - 2][2], ptsData[index - 2][3],
                             con,
                             -1])
        else:
            index = oneSplit[-1][5]
            oneSplit.insert(0,
                            [ptsData[index - 1][0], ptsData[index - 1][1], ptsData[index - 1][2], ptsData[index - 1][3],
                             con,
                             -1])
            oneSplit.append(
                [ptsData[index + 1][0], ptsData[index + 1][1], ptsData[index + 1][2], ptsData[index + 1][3], con,
                 -1])

    return oneSplit


def splitpixelsFeatures(ptsSplit, ptsData, label):
    Speedlist = []
    absDeltaSlst = []
    absDelTurnlst = []

    for q in range(len(ptsSplit)):
        if len(ptsSplit[q]) >= 3:
            valueTemp = calAttr(ptsSplit[q], label)
            Speedlist.append(valueTemp[0])
            absDeltaSlst.append(valueTemp[1])
            absDelTurnlst.append(valueTemp[2])

        elif 3 > len(ptsSplit[q]) >= 2:
            Speedlist.append(calSpeed(ptsSplit[q], label))
            ptsFill = fillPts(ptsSplit[q], ptsData, 1)
            valueTemp = calAttr(ptsFill, label)
            absDeltaSlst.append(valueTemp[1])
            absDelTurnlst.append(valueTemp[2])

        elif 2 > len(ptsSplit[q]) >= 1:
            ptsFill = fillPts(ptsSplit[q], ptsData, 1)
            Speedlist.append(calSpeed(ptsFill, label))

            ptsFill = fillPts(ptsSplit[q], ptsData, 2)
            valueTemp = calAttr(ptsFill, label)
            absDeltaSlst.append(valueTemp[1])
            absDelTurnlst.append(valueTemp[2])

    Speedlist = list(_flatten(Speedlist))
    absDeltaSlst = list(_flatten(absDeltaSlst))
    absDelTurnlst = list(_flatten(absDelTurnlst))

    # if not Speedlist:
    #     Speedlist.append(0)
    # if not absDeltaSlst:
    #     absDeltaSlst.append(0)
    #     absDelTurnlst.append(0)

    fearesult = [np.mean(Speedlist), np.max(Speedlist), np.min(Speedlist),
                 np.mean(absDeltaSlst), np.max(absDeltaSlst), np.min(absDeltaSlst),
                 np.mean(absDelTurnlst), np.max(absDelTurnlst), np.min(absDelTurnlst)]
    return fearesult


def calpixelsFeatures(imgPtsData, pts, modes):
    print("------calpiexelsFeatures START---------")
    AveSImgs, MaxSImgs, MedianSImgs = [], [], []
    AveAbsDelSImgs, MaxAbsDelSImgs, MedianAbsDelSImgs = [], [], []
    aveAbsDelTurnImgs, maxAbsDelTurnImgs, medianAbsDelTurnImgs = [], [], []

    for i in range(len(imgPtsData)):
        mode = modes[i]
        AveSImg5, MaxSImg5, MedianSImg5 = [], [], []
        AveAbsDelSImg5, MaxAbsDelSImg5, MedianAbsDelSImg5 = [], [], []
        aveAbsDelTurnImg5, maxAbsDelTurnImg5, medianAbsDelTurnImg5 = [], [], []
        for v in range(len(imgPtsData[i])):
            oneImgPts = imgPtsData[i][v]
            onePts = pts[i]

            AveSImg = [[0 for q in range(Wm)] for t in range(Hm)]
            MaxSImg = [[0 for q in range(Wm)] for t in range(Hm)]
            MedianSImg = [[0 for q in range(Wm)] for t in range(Hm)]

            AveAbsDelSImg = [[0 for q in range(Wm)] for t in range(Hm)]
            MaxAbsDelSImg = [[0 for q in range(Wm)] for t in range(Hm)]
            MedianAbsDelSImg = [[0 for q in range(Wm)] for t in range(Hm)]

            aveAbsDelTurnImg = [[0 for q in range(Wm)] for t in range(Hm)]
            maxAbsDelTurnImg = [[0 for q in range(Wm)] for t in range(Hm)]
            medianAbsDelTurnImg = [[0 for q in range(Wm)] for t in range(Hm)]

            for j in range(len(oneImgPts)):
                for k in range(len(oneImgPts[j])):
                    onePixel = oneImgPts[j][k]
                    if len(onePixel) > 0:
                        onePixelSp = splitPixelsPts(onePixel)
                        if len(onePixelSp) >= 1:
                            features = splitpixelsFeatures(onePixelSp, onePts, mode)

                            AveSImg[j][k] = features[0]
                            MaxSImg[j][k] = features[1]
                            MedianSImg[j][k] = features[2]

                            AveAbsDelSImg[j][k] = features[3]
                            MaxAbsDelSImg[j][k] = features[4]
                            MedianAbsDelSImg[j][k] = features[5]

                            aveAbsDelTurnImg[j][k] = features[6]
                            maxAbsDelTurnImg[j][k] = features[7]
                            medianAbsDelTurnImg[j][k] = features[8]

                    else:
                        continue
            AveSImg5.append(AveSImg)
            MaxSImg5.append(MaxSImg)
            MedianSImg5.append(MedianSImg)

            AveAbsDelSImg5.append(AveAbsDelSImg)
            MaxAbsDelSImg5.append(MaxAbsDelSImg)
            MedianAbsDelSImg5.append(MedianAbsDelSImg)  

            aveAbsDelTurnImg5.append(aveAbsDelTurnImg)
            maxAbsDelTurnImg5.append(maxAbsDelTurnImg)
            medianAbsDelTurnImg5.append(medianAbsDelTurnImg)

        AveSImgs.append(AveSImg5)
        MaxSImgs.append(MaxSImg5)
        MedianSImgs.append(MedianSImg5)

        AveAbsDelSImgs.append(AveAbsDelSImg5)
        MaxAbsDelSImgs.append(MaxAbsDelSImg5)
        MedianAbsDelSImgs.append(MedianAbsDelSImg5)  

        aveAbsDelTurnImgs.append(aveAbsDelTurnImg5)
        maxAbsDelTurnImgs.append(maxAbsDelTurnImg5)
        medianAbsDelTurnImgs.append(medianAbsDelTurnImg5)

    pixelsFea = [AveSImgs, MaxSImgs, MedianSImgs, AveAbsDelSImgs,
                 MaxAbsDelSImgs, MedianAbsDelSImgs, aveAbsDelTurnImgs, maxAbsDelTurnImgs, medianAbsDelTurnImgs]  

    print("------calpiexelsFeatures DONE---------")
    return pixelsFea


def calInput(features):
    print("------calculate Input START---------")
    inputResult = []
    AS = features[0]
    maxS = features[1]
    medianS = features[2]
    AAbsDelS = features[3]
    maxAbsDelS = features[4]
    medianAbsDelS = features[5]
    AAbsDelTR = features[6]
    maxAbsDelTR = features[7]
    medianAbsDelTR = features[8]
    for i in range(len(features[0])):
        temp = []
        temp.append([AS[i][0], maxS[i][0], medianS[i][0], AAbsDelS[i][0],
                     maxAbsDelS[i][0], medianAbsDelS[i][0], AAbsDelTR[i][0], maxAbsDelTR[i][0],
                     medianAbsDelTR[i][0]])  #
        temp.append([AS[i][1], maxS[i][1], medianS[i][1], AAbsDelS[i][1],
                     maxAbsDelS[i][1], medianAbsDelS[i][1], AAbsDelTR[i][1], maxAbsDelTR[i][1], medianAbsDelTR[i][1]])
        temp.append([AS[i][2], maxS[i][2], medianS[i][2], AAbsDelS[i][2],
                     maxAbsDelS[i][2], medianAbsDelS[i][2], AAbsDelTR[i][2], maxAbsDelTR[i][2], medianAbsDelTR[i][2]])
        temp.append([AS[i][3], maxS[i][3], medianS[i][3], AAbsDelS[i][3],
                     maxAbsDelS[i][3], medianAbsDelS[i][3], AAbsDelTR[i][3], maxAbsDelTR[i][3], medianAbsDelTR[i][3]])
        temp.append([AS[i][4], maxS[i][4], medianS[i][4], AAbsDelS[i][4],
                     maxAbsDelS[i][4], medianAbsDelS[i][4], AAbsDelTR[i][4], maxAbsDelTR[i][4], medianAbsDelTR[i][4]])

        # temp.append([AS[i][5], maxS[i][5], medianS[i][5], AAbsDelS[i][5],
        #              maxAbsDelS[i][5], medianAbsDelS[i][5], AAbsDelTR[i][5], maxAbsDelTR[i][5], medianAbsDelTR[i][5]])
        # temp.append([AS[i][6], maxS[i][6], medianS[i][6], AAbsDelS[i][6],
        #              maxAbsDelS[i][6], medianAbsDelS[i][6], AAbsDelTR[i][6], maxAbsDelTR[i][6], medianAbsDelTR[i][6]])
        # temp.append([AS[i][7], maxS[i][7], medianS[i][7], AAbsDelS[i][7],
        #              maxAbsDelS[i][7], medianAbsDelS[i][7], AAbsDelTR[i][7], maxAbsDelTR[i][7], medianAbsDelTR[i][7]])
        # temp.append([AS[i][8], maxS[i][8], medianS[i][8], AAbsDelS[i][8],
        #              maxAbsDelS[i][8], medianAbsDelS[i][8], AAbsDelTR[i][8], maxAbsDelTR[i][8], medianAbsDelTR[i][8]])

        inputResult.append(temp)

    print("------calculate Input DONE---------")
    return inputResult


allSegmentImage, imgModes, allSegment, modes  # variables form step 3and 4

pixelFeature = calpixelsFeatures(allSegmentImage, allSegment, imgModes)
imgFeatures = calInput(pixelFeature)
# Out imgFeatures
