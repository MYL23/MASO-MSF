import torch
import numpy as np

OBJECT_K = 6


def confusionMatrix(matrix, preds, labels):
    for p, q in zip(labels, preds):
        matrix[p, q] += 1
    return matrix


def takeSecond(elem):
    return elem[1]


def getMaxProbIndex(probData):
    probData.sort(key=takeSecond)
    return probData[-1][0]


def getFullMaxIndex(probData):
    return np.argmax(probData)


def getRepeat(dict):
    repeatIndex = []
    tempIndex = []
    for key in dict:
        if len(dict[key]) == 2:
            tempIndex.append(dict[key])
    for p in range(len(tempIndex)):
        repeatIndex.append(tempIndex[p][0])
        repeatIndex.append(tempIndex[p][1])
    return repeatIndex


def getsubProb(index, probData):
    result = []
    for i in range(len(index)):
        result.append([index[i], probData[index[i]]])
    return result


def getFinalPred(lst, prob):
    countFreq = {}
    for key in lst:
        countFreq[key] = countFreq.get(key, 0) + 1
    highest = max(countFreq.values())  # 3

    a = [k for k, v in countFreq.items() if v == highest]  # [0,1]
    if len(a) == 1:
        truelabel = max(lst, default='列表为空', key=lambda v: lst.count(v))
    elif len(a) == OBJECT_K:
        truelabel = lst[getFullMaxIndex(prob)]
    else:
        sumpro = {}
        for p in range(len(a)):
            b = [v for k, v in enumerate(prob) if lst[k] == a[p]]
            sumpro[a[p]] = np.mean(b)
        truelabel = max(sumpro, key=sumpro.get)

    return truelabel


def ObjectDecisionFusion(data):
    predictlabel = data.argmax(dim=1).tolist()
    probability = torch.max(data, 1)[0].tolist()
    k = 0
    maxLabel = []
    while k < len(predictlabel):
        tempLabel = predictlabel[k:k + OBJECT_K]
        tempProb = probability[k:k + OBJECT_K]
        maxLabel.append(getFinalPred(tempLabel, tempProb))
        k += OBJECT_K
    maxLabel = torch.tensor(maxLabel).type(torch.LongTensor)

    return maxLabel


def getExpandLabel(data):
    data = data.tolist()
    y_convert = []
    for k in range(len(data)):
        for p in range(OBJECT_K):
            y_convert.append(data[k])
    y_convert = torch.tensor(y_convert).type(torch.LongTensor)
    return y_convert


def testModel(net, test_iter, criterion, device, condition, num_classes):
    total, correct = 0, 0
    lossValue = 0
    net.eval()
    confMatrixs = []

    with torch.no_grad():
        for step, (X1, X2, X3, X4, y) in enumerate(
                test_iter):  #
            X1, X2, X3, X4, y = X1.to(device), X2.to(device), X3.to(device), X4.to(device), y.to(device)
            output = net(X1, X2, X3, X4)
            y_convert = getExpandLabel(y).to(device)

            loss = criterion(output, y_convert)
            lossValue += loss.data.item()
            total += y.size(0)
            maxLabel = ObjectDecisionFusion(output)
            y = y.cpu()
            correctNum = (maxLabel == y).sum().item()
            correct += correctNum
            if step == 0:
                confMatrixs = torch.zeros(num_classes, num_classes)
            if condition == "test":
                confMatrixs = confusionMatrix(confMatrixs, maxLabel, y)

        test_acc = 100.0 * correct / total
        test_loss = lossValue / len(test_iter)
        if condition == "train":
            print("train_loss: {:.3f} | train_acc: {:6.3f}% |" \
                  .format(test_loss, test_acc))
        elif condition == "test":
            print("test_loss: {:.3f} | test_acc: {:6.3f}% | " \
                  .format(test_loss, test_acc))

    net.train()

    return (test_loss, test_acc, confMatrixs)
