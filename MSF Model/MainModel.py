import torch
from torch import nn
import numpy as np
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBJECT_K = 6


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio):
        super(ChannelAttention, self).__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, 1, 1, 0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel // ratio, in_channel, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_sq = self.squeeze(x)
        x_weight = self.excitation(x_sq)
        out = x * x_weight
        return out


class ResidualBlock(nn.Module):
    def __init__(self, nin, nout, size, stride=1, shortcut=True, ratioValue=16):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=size, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nout, nout, kernel_size=size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nout))
        self.shortcut = shortcut
        self.block2 = nn.Sequential(
            nn.Conv2d(nin, nout, size, stride, 1, bias=False),
            nn.BatchNorm2d(nout),
        )
        self.attention = ChannelAttention(nout, ratioValue)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input_x):
        x = input_x
        out = self.block1(x)
        out = self.attention(out)
        if self.shortcut:
            out = x + out
        else:
            out = out + self.block2(x)
        out = self.relu(out)
        return out


class AttrChannelFusion(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, ratio):
        super(AttrChannelFusion, self).__init__()
        self.ShallowConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel1),
            nn.LeakyReLU(),
        )
        self.d1 = self._make_layers(out_channel1, out_channel1, 3, stride=1, t=1, ratio_value=ratio)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.d2 = self._make_layers(out_channel1, out_channel2, 3, stride=1, t=1, ratio_value=ratio)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

    def _make_layers(self, in1, out1, ksize, stride, t, ratio_value):
        layers = []
        for i in range(0, t):
            if i == 0 and in1 != out1:
                layers.append(ResidualBlock(in1, out1, ksize, stride, None, ratio_value))
            else:
                layers.append(ResidualBlock(out1, out1, ksize, 1, True, ratio_value))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_conv = self.ShallowConv(x)
        x_d1 = self.pool1(self.d1(x_conv))
        x_d2 = self.pool2(self.d2(x_d1))
        return x_d2


def processSamples(dataSample):
    return dataSample.contiguous().view(-1, dataSample.shape[2], dataSample.shape[3],
                                        dataSample.shape[4])


valueFile = r'/..'  # normalized parameters for MASO data: mean and standard deviation
with open(valueFile, 'rb') as f:
    normalizeValue = pickle.load(f)


def normlize(data, value):  
    dataArray = data.cpu().numpy()
    for k in range(len(dataArray[0])):
        condlist = [dataArray[:, k, :, :] != 0]
        choicelist = [(dataArray[:, k, :, :] - value[k][0]) / value[k][1]]  # [0]:mean [1]:stand deviation
        dataArray[:, k, :, :] = np.select(condlist, choicelist)
        del condlist, choicelist
    result = torch.from_numpy(dataArray).to(DEVICE)
    return result


class ScaleAttention(nn.Module):
    def __init__(self):
        super(ScaleAttention, self).__init__()
        self.FCLayer = nn.Sequential(
            nn.Linear(128 * 3, 128 * 3),
            nn.BatchNorm1d(128 * 3),
            nn.Tanh()
        )
        self.Activation = nn.Sequential(
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_convert = self.FCLayer(x)
        scale_weight = self.Activation(x_convert)
        x_weight = x * scale_weight
        return x_weight


def processGlobal(globalfeatures):  # share the global attributes
    expandGlobal = []
    globaldata = globalfeatures.cpu().tolist()
    for k in range(len(globaldata)):
        for j in range(OBJECT_K):
            expandGlobal.append(globaldata[k])
    expandGlobal = torch.tensor(expandGlobal).type(torch.FloatTensor).to(DEVICE)
    return expandGlobal


class mergeScales(nn.Module):
    def __init__(self, in_channel_scale1, in_channel_scale2, in_channel_scale3, classes, ratio):
        super(mergeScales, self).__init__()
        self.moduleForScale1 = AttrChannelFusion(in_channel=in_channel_scale1, out_channel1=32, out_channel2=64,
                                                 ratio=ratio)  # 0.005*32
        self.moduleForScale2 = AttrChannelFusion(in_channel=in_channel_scale2, out_channel1=32, out_channel2=64,
                                                 ratio=ratio)  # 0.05
        self.moduleForScale3 = AttrChannelFusion(in_channel=in_channel_scale3, out_channel1=32, out_channel2=64,
                                                 ratio=ratio)  # 0.1
        self.FCLayerForOut = nn.Sequential(
            nn.Linear(128 * 3 + 36, 32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, classes)
        )
        self.FCLayerForScale1 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512), 
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128),  
            nn.LeakyReLU(inplace=True),
        )
        self.FCLayerForScale2 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),  
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128),  
            nn.LeakyReLU(inplace=True),
        )
        self.FCLayerForScale3 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),  
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128),  
            nn.LeakyReLU(inplace=True),
        )
        self.attentionForScales = ScaleAttention()

    def forward(self, x1, x2, x3, x4):
        # For scale1:ACFM
        input1 = normlize(processSamples(x1), normalizeValue)
        output1 = self.moduleForScale1(input1)

        # For scale2:ACFM
        input2 = normlize(processSamples(x2), normalizeValue)
        output2 = self.moduleForScale2(input2)

        # For scale3:ACFM
        input3 = normlize(processSamples(x3), normalizeValue)
        output3 = self.moduleForScale3(input3)

        # SFFM:Scale Feature Fusion Module
        output1 = output1.view(output1.shape[0], -1)
        output1 = self.FCLayerForScale1(output1)

        output2 = output2.view(output2.shape[0], -1)
        output2 = self.FCLayerForScale2(output2)

        output3 = output3.view(output3.shape[0], -1)
        output3 = self.FCLayerForScale3(output3)

        scale_fusion = torch.cat([output1, output2, output3], dim=1)
        scale_weight_fusion = self.attentionForScales(scale_fusion)

        # Additional Global Attributes Fusion
        global_attr = processGlobal(x4)
        attr_fusion = torch.cat([scale_weight_fusion, global_attr], dim=1)

        # finalout
        out_fusion = self.FCLayerForOut(attr_fusion)

        return out_fusion

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


def MASO_MSF(num_channel, num_classes, value_ratio):
    return mergeScales(in_channel_scale1=num_channel, in_channel_scale2=num_channel, in_channel_scale3=num_channel,
                       classes=num_classes, ratio=value_ratio)
