# step 3: segment cropping and project segment to img

import os
import numpy as np
import pickle
import math
import gc

os.chdir('/..')

# scale definition
Wp = Hp = 0.2  # spatial range
Wm = Hm = 32  # image size
parts = 6  # number of objects


def getSubSeg(data):
    length = len(data)
    sublength = round(length / parts)
    subIndex = []
    for k in range(parts):
        if k != parts - 1:
            subIndex.append(math.floor((k + 0.5) * sublength))
        else:
            subIndex.append(math.floor((length - k * sublength) * 0.5 + k * sublength))
    if subIndex[-1] == length:
        subIndex[-1] = length - 1
    return subIndex


def convert2Img(segData):
    imgPoints = []
    print("------convert START---------")
    for i in range(len(segData)):
        oneSample = segData[i]
        imgPoint = []
        centerIndex = getSubSeg(oneSample)
        for v in range(len(centerIndex)):
            index = centerIndex[v]
            centerLat = oneSample[index][0]
            centerLng = oneSample[index][1]

            minLng = np.min(oneSample, 0)[1]
            minLat = np.min(oneSample, 0)[0]

            offsetX = math.floor(Wm / 2) - math.floor((centerLng - minLng) * Wm / Wp)
            offsetY = math.floor(Hm / 2) - math.floor((centerLat - minLat) * Hm / Hp)


            oneSegmentImage = [[[] for p in range(Wm)] for t in range(Hm)]
            con = [[0 for p in range(Wm)] for t in range(Hm)]
            recordIndex = [[0 for p in range(Wm)] for t in range(Hm)]

            for k in range(len(oneSample)):
                imageX = math.floor((oneSample[k][1] - minLng) * Wm / Wp) + offsetX
                imageY = math.floor((oneSample[k][0] - minLat) * Hm / Hp) + offsetY

                if (0 <= imageX < Wm) and (0 <= imageY < Hm):
                    indexX = Wm - 1 - imageY
                    indexY = imageX
                    if k == recordIndex[indexX][indexY] + 1:  # the GPS points in the same pixel are consecutive?
                        insertPoint = [oneSample[k][0], oneSample[k][1], oneSample[k][2], oneSample[k][3],
                                       con[indexX][indexY], k]
                        oneSegmentImage[indexX][indexY].append(insertPoint)
                        recordIndex[indexX][indexY] = k
                    else:
                        con[indexX][indexY] += 1
                        insertPoint = [oneSample[k][0], oneSample[k][1], oneSample[k][2], oneSample[k][3],
                                       con[indexX][indexY], k]
                        oneSegmentImage[indexX][indexY].append(insertPoint)
                        recordIndex[indexX][indexY] = k
            imgPoint.append(oneSegmentImage)
        imgPoints.append(imgPoint)
    print("------convert DONE---------")
    return imgPoints



allSegment #variable from step 2
allSegmentImage = convert2Img(allSegment)

# Output allSegmentImage


