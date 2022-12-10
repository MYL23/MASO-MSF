import torch
import numpy as np
import os
import pickle
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import gc
import re
import torchvision.transforms as transforms


class myDataSet(Data.Dataset):
    def __init__(self, files, transform=None):
        self.dataFiles = files
        self.transform = transform

    def __getitem__(self, index):
        file = open(self.dataFiles[index], 'rb')
        data = pickle.load(file)
        # 处理
        data1 = torch.tensor(data[0]).type(torch.FloatTensor)  # scale1 [n*object_k*num_channel*Wm*Hm]
        data2 = torch.tensor(data[1]).type(torch.FloatTensor)  # scale2
        data3 = torch.tensor(data[2]).type(torch.FloatTensor)  # scale3
        data4 = torch.tensor(data[3]).type(torch.FloatTensor)  # global attributes

        target = torch.tensor(data[4]).type(torch.LongTensor)
        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
            data3 = self.transform(data3)
        return data1, data2, data3, data4, target

    def __len__(self):
        return len(self.dataFiles)


def getIndexOfFiles(root, indexs):
    dataFiles = np.array([x.path for x in os.scandir(root)])
    sortFiles = sorted(dataFiles,
                       key=lambda x: int(re.findall("\d+", x.split("/")[-1].split(".pickle")[0])[0]))
    indexOfFiles = [q for p, q in enumerate(sortFiles) if p in indexs]

    return indexOfFiles


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
])


def load_dataset(batch_size):
    # print("------- data START---------")
    filename = r'/home/kwan30902/Workspace/myl/multiAttrScale/Data/mediumFiles/SegmentsPtsModes.pickle'
    with open(filename, 'rb') as f:
        pts, modes = pickle.load(f)
    del pts
    gc.collect()

    index = [i for i in range(len(modes))]
    trainX_index, testX_index, trainY, testY = train_test_split(index, modes, test_size=0.2, random_state=1,
                                                                stratify=modes)  # base on modes,for balance

    sample_roots = r'/home/kwan30902/Workspace/myl/multiAttrScale/Data/mediumFiles/samples/6M32GF/'
    train_index_files = getIndexOfFiles(sample_roots, trainX_index)
    test_index_files = getIndexOfFiles(sample_roots, testX_index)
    train_set = myDataSet(train_index_files, transform_train)
    train_iter = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                                 num_workers=4)

    test_set = myDataSet(test_index_files)
    test_iter = Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # print("----- Data Done----------")
    return train_iter, test_iter
