# step 2: split trajectory to segments
import os
import numpy as np
import pickle

os.chdir('/..')
filename = r'/..'# from the first step
AllSegment = []
AllModes = []
minPoints = 20
AllsegmentPointsNumber = []

trajectoryLabelAllUser #variable from step 1

for t in range(len(trajectoryLabelAllUser)):
    Data = trajectoryLabelAllUser[t]
    if len(Data) == 0:
        continue
    delta_time = []

    for i in range(len(Data) - 1):
        delta_time.append((Data[i + 1, 2] - Data[i, 2]) * 24. * 3600)
        if delta_time[i] == 0:
            delta_time[i] = 0.1
        A = (Data[i, 0], Data[i, 1])
        B = (Data[i + 1, 0], Data[i + 1, 1])
    delta_time.append(3)

    min_trip_time = 20 * 60  # zheng et al., 2008
    tripPointsNumber = []
    dataOneUserTrip = []
    dataOneUserSegment = []
    counter = 0
    index = []

    i = 0
    while i <= (len(Data) - 1):
        if delta_time[i] <= min_trip_time:
            counter += 1
            index.append(i)
            i += 1
        else:
            counter += 1
            index.append(i)
            i += 1

            tripPointsNumber.append(counter)
            dataTrip = [Data[k, 0:4] for k in index]
            dataTrip = np.array(dataTrip, dtype=float)
            dataOneUserTrip.append(dataTrip)
            counter = 0
            index = []
            continue
    if len(index) != 0:
        tripPointsNumber.append(counter)
        dataTrip = [Data[k, 0:4] for k in index]
        dataTrip = np.array(dataTrip, dtype=float)
        dataOneUserTrip.append(dataTrip)
    else:
        print("last trip ")

    i = 0
    for i in range(len(dataOneUserTrip)):

        dataTrip = dataOneUserTrip[i]
        modeType = dataTrip[0][3]
        index = []

        j = 0
        while j <= len(dataTrip) - 1:

            if dataTrip[j][3] == modeType:
                index.append(j)
                j += 1
            else:
                dataSegment = [dataTrip[k, 0:4] for k in index]
                dataOneUserSegment.append(dataSegment)
                index = []
                modeType = dataTrip[j][3]
                index.append(j)
                j += 1
                continue
        if len(index) != 0:
            dataSegment = [dataTrip[k, 0:4] for k in index]
            dataOneUserSegment.append(dataSegment)
        else:
            print("last segment")

    segmentPointsNumber = []
    for i in range(len(dataOneUserSegment)):
        if len(dataOneUserSegment[i]) >= minPoints:
            segmentPointsNumber.append(len(dataOneUserSegment[i]))
            AllsegmentPointsNumber.append(len(dataOneUserSegment[i]))
            AllSegment.append(dataOneUserSegment[i])
            AllModes.append(int(dataOneUserSegment[i][0][3]))
    if len(segmentPointsNumber) == 0:
        continue
#Output
with open('...pickle', 'wb') as f:
    pickle.dump([AllSegment, AllModes], f)
