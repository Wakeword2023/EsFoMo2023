"""
Imported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/densenet.py
and added support for the 1x32x32 mel spectrogram for the speech recognition.

Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten: Densely Connected Convolutional Networks
https://arxiv.org/abs/1608.06993
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

from torch.autograd import Variable

__all__ = [ 'DenseNet' ]

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn1Name = str(self.bn1)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1Name = str(self.conv1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2Name = str(self.bn2)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.conv2Name = str(self.conv2)
        self.relu = nn.ReLU(inplace=True)
        self.reluName = str(self.relu)
        self.dropRate = dropRate
        self.names = []
        self.times = []

    def forward(self, x):
        self.names = []
        self.times = []
        
        self.names.append(self.bn1Name)
        currentTime = time.time()
        out = self.bn1(x)
        self.times.append(time.time() - currentTime)

        self.names.append(self.reluName)
        currentTime = time.time()
        out = self.relu(out)
        self.times.append(time.time() - currentTime)

        self.names.append(self.conv1Name)
        currentTime = time.time()
        out = self.conv1(out)
        self.times.append(time.time() - currentTime)

        self.names.append(self.bn2Name)
        currentTime = time.time()
        out = self.bn2(out)
        self.times.append(time.time() - currentTime)

        self.names.append(self.reluName)
        currentTime = time.time()
        out = self.relu(out)
        self.times.append(time.time() - currentTime)

        self.names.append(self.conv2Name)
        currentTime = time.time()
        out = self.conv2(out)
        self.times.append(time.time() - currentTime)

        if self.dropRate > 0:
            
            out = F.dropout(out, p=self.dropRate, training=self.training)

        else:
            v = time.time()

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn1Name = str(self.bn1)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.conv1Name = str(self.conv1)
        self.relu = nn.ReLU(inplace=True)
        self.reluName = str(self.relu)
        self.dropRate = dropRate

        self.names = []
        self.times = []

    def forward(self, x):
        self.names = []
        self.times = []


        self.names.append(self.bn1Name)
        currentTime = time.time()
        out = self.bn1(x)
        self.times.append(time.time() - currentTime)


        self.names.append(self.reluName)
        currentTime = time.time()
        out = self.relu(out)
        self.times.append(time.time() - currentTime)


        self.names.append(self.conv1Name)
        currentTime = time.time()
        out = self.conv1(out)
        self.times.append(time.time() - currentTime)

        if self.dropRate > 0:
            self.names.append("F.dropout(out, p=self.dropRate, training=self.training)")
            currentTime = time.time()
            out = F.dropout(out, p=self.dropRate, training=self.training)
            self.times.append(time.time() - currentTime)
        

        self.names.append("torch.cat((x, out), 1)")
        currentTime = time.time()
        out = torch.cat((x, out), 1)
        self.times.append(time.time() - currentTime)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn1Name = str(self.bn1)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.conv1Name = str(self.conv1)
        self.relu = nn.ReLU(inplace=True)
        self.reluName = str(self.relu)
        
        self.names = []
        self.times = []
        

    def forward(self, x):
        self.names = []
        self.times = []
        

        self.names.append(self.bn1Name)
        currentTime = time.time()
        out = self.bn1(x)
        self.times.append(time.time() - currentTime)
        

        self.names.append(self.reluName)
        currentTime = time.time()
        out = self.relu(out)
        self.times.append(time.time() - currentTime)


        self.names.append(self.conv1Name)
        currentTime = time.time()
        out = self.conv1(out)
        self.times.append(time.time() - currentTime)


        self.names.append("F.avg_pool2d(out, 2)")
        currentTime = time.time()
        out = F.avg_pool2d(out, 2)
        self.times.append(time.time() - currentTime)


        return out


class DenseNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck,
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2, in_channels=3):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate
        self.currentTime= time.time()

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.conv1Name = str(self.conv1)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.bnName = str(nn.BatchNorm2d(self.inplanes))
        self.relu = nn.ReLU(inplace=True)
        self.reluName = str(self.relu)
        self.avgpool = nn.AvgPool2d(8)
        self.avgpoolName = str(self.avgpool)
        self.fc = nn.Linear(self.inplanes, num_classes)
        self.fcName = str(self.fc)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.name = "DenseNet"
        self.names = []
        self.times = []

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)


    def forward(self, x):
        self.names = []
        self.times = []

        self.names.append(self.conv1Name)
        currentTime = time.time()
        x = self.conv1(x)
        self.times.append(time.time() - currentTime)
        currentTime = time.time()


        for i in range(0, len(self.dense1)):
            self.names.extend(self.dense1[i].names)
            x = (self.dense1[i])(x)
            self.times.extend(self.dense1[i].times)

        self.names.extend(self.trans1.names)
        x = (self.trans1)(x)
        self.times.extend(self.trans1.times)

        for i in range(0, len(self.dense2)):
            self.names.extend(self.dense2[i].names)
            x = (self.dense2[i])(x)
            self.times.extend(self.dense2[i].times)

        self.names.extend(self.trans2.names)
        x = (self.trans2)(x)
        self.times.extend(self.trans2.times)

        for i in range(0, len(self.dense3)):
            self.names.extend(self.dense3[i].names)
            x = (self.dense3[i])(x)
            self.times.extend(self.dense3[i].times)

        self.names.append(self.bnName)
        currentTime = time.time()
        x = self.bn(x)
        self.times.append(time.time() - currentTime)
        currentTime = time.time()

        self.names.append(self.reluName)
        currentTime = time.time()
        x = self.relu(x)
        self.times.append(time.time() - currentTime)
        currentTime = time.time()
  
        self.names.append(self.avgpoolName)
        currentTime = time.time()
        x = self.avgpool(x)
        self.times.append(time.time() - currentTime)
        currentTime = time.time()
        
        x = x.view(x.size(0), -1)

        self.names.append(self.fcName)
        currentTime = time.time()
        x = self.fc(x)
        self.times.append(time.time() - currentTime)
        currentTime = time.time()

        return x
