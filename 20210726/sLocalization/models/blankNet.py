"""
@author: Kojungbeom

"""

import torch
import torch.nn as nn


class NormBlock(nn.Module):
    """Our special unit
    """
    def __init__(self):
        super().__init__()
        self.mp = nn.MaxPool2d((3,1), stride=(2,1), padding=(1,0))
        self.num_classes = 8
    def forward(self, x):
        out = self.mp(x)
        norm_x = out.clone()
        for i in range(len(x)):
            max_x = torch.max(norm_x[i])
            min_x = torch.min(norm_x[i])
            norm_x[i] = (norm_x[i]-min_x) / (max_x - min_x)
        return norm_x


class SpecialBlockA(nn.Module):
    """Our special unit
    """
    def __init__(self, in_channels, out_channels, f_size):
        super().__init__()
        self.num_classes = 7
        self.low_k = 3
        self.middle_k = 6
        self.high_k = 12

        self.block_low = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.low_k, 8),
                      stride=(int(self.low_k / 2),1),
                      padding=(round(self.low_k / 4 + 0.5), 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d((6,1))
        )

        self.block_middle = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.middle_k, 8),
                      stride=(int(self.middle_k / 2), 1),
                      padding=(round(self.middle_k / 4 + 0.5), 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d((2,1))
        )

        self.block_high = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.high_k, 8),
                      stride=(int(self.high_k / 2), 1),
                      padding=(round(self.high_k / 4 + 0.5), 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d((1,1))
        )


    def forward(self, x):
        out1 = self.block_low(x)
        #print(out1.shape)
        out2 = self.block_middle(x)
        #print(out2.shape)
        out3 = self.block_high(x)
        #print(out3.shape)
        #return out1
        return torch.cat((out1, out2, out3), 1)

class BlankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 8
        self.outc = 64
        self.inplanes = 1
        self.norm = NormBlock()
        self.layer1 = nn.Sequential(SpecialBlockA(1, self.outc, 5))
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 1))
        self.fc_layer = nn.Sequential(
            nn.Linear(self.outc*4*3, self.num_classes)
        )

    def forward(self, x):
        out = self.norm(x)
        out = self.layer1(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc_layer(out)
        return out
