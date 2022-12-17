import numpy as np
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig2 = plt.figure(2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xlabel('Predicted Mode')
    plt.ylabel('Real Mode')
    plt.show()


def getPRF1(confmatirx, num_classes):
    print("\n (2)Confusion Matrix:\n", confmatirx)
    confmatirx = confmatirx.numpy()

    types = ['walk', 'bike', 'bus', 'driving', 'train']
    precision = [0 for x in range(num_classes)]
    recall = [0 for x in range(num_classes)]
    F1Score = [0 for x in range(num_classes)]
    correct = 0
    for i in range(len(confmatirx)):
        for j in range(len(confmatirx[i])):
            if i == j:
                recall[i] = confmatirx[i][j] / np.sum(confmatirx, -1)[i]
                precision[i] = confmatirx[i][j] / np.sum(confmatirx, 0)[i]  
                correct += confmatirx[i][j]
    print("(3) the precision:", precision)
    print("(4) the recall:", recall)
    for i in range(len(precision)):
        F1Score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
    print("(5) the F1-SCORE", F1Score)

    # plot the matirx
    plot_confusion_matrix(confmatirx, classes=types, normalize=False, title="confusion matrix")


def printMetrics(test_loss_acc, confusion, num_classes):
    test_acc = [x[1] for x in test_loss_acc]

    max_index = test_acc.index(max(test_acc))
    print("\n(1) the optimal epoch", max_index + 1, ", the best identification accuracy:", max(test_acc))
    best_confusion = confusion[max_index]
    getPRF1(best_confusion, num_classes)
