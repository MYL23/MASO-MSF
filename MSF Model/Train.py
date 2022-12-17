import torch
from torch import nn
import os
import time
from tensorboardX import SummaryWriter
from MainModel import MASO_MSF
from DataLoader import load_dataset
from PrintMetricInformation import printMetrics
from Test import testModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBJECT_K = 6

os.chdir(r'/..')


def getExpandLabel(data):
    data = data.tolist()
    y_convert = []
    for k in range(len(data)):
        for p in range(OBJECT_K):
            y_convert.append(data[k])
    y_convert = torch.tensor(y_convert).type(torch.LongTensor)
    return y_convert


def trainModel(net, train_iter, test_iter, criterion, optimizer, num_epochs, device, num_classes):
    print("----- train START----------")
    net.train()
    record_train = []
    record_test = []
    confusion = []

    for epoch in range(num_epochs):
        start = time.time()
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        for i, (X1, X2, X3, X4, y) in enumerate(
                train_iter):
            optimizer.zero_grad()
            X1, X2, X3, X4, y = X1.to(device), X2.to(device), X3.to(device), X4.to(device), y.to(device)
            output = net(X1, X2, X3, X4)
            y_convert = getExpandLabel(y).to(device)
            loss = criterion(output, y_convert)

            loss.backward()
            optimizer.step()

        print("--- training cost time: {:.4f}s ---".format(time.time() - start))

        # each epoch calculate the accuracy and loss of train and test
        dict_name = 'MASOMSF-' + str(epoch + 1) + '.pt'
        torch.save(net.state_dict(), dict_name)

        t1 = testModel(net, train_iter, criterion, device, "train", num_classes)
        t2 = testModel(net, test_iter, criterion, device, "test", num_classes)
        record_train.append([t1[0], t1[1]])
        record_test.append([t2[0], t2[1]])
        confusion.append(t2[2])

        print("--- cost time: {:.4f}s ---".format(time.time() - start))

    return record_train, record_test, confusion


def main():
    # parameters
    num_classes = 5
    num_channel = 9
    value_ratio = 16  # channel attention
    learning_rate = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-08
    decay = 0.0
    BATCH_SIZE = 16
    NUM_EPOCHS = 3

    # model
    if torch.cuda.is_available():
        print("cuda true")

    net = MASO_MSF(num_channel, num_classes, value_ratio)
    net.initialize_weight()
    net = net.to(DEVICE)

    # loadData
    train_iter, test_iter = load_dataset(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=epsilon,
                                 weight_decay=decay)

    record_train, record_test, confusion_data = trainModel(net, train_iter, test_iter, criterion, optimizer, NUM_EPOCHS,
                                                           DEVICE, num_classes)
    printMetrics(record_test, confusion_data, num_classes)


if __name__ == '__main__':
    main()
