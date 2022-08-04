"""
@author: Kojungbeom

"""

import torch
import torch.nn as nn

class SpecialBlock2(nn.Module):
    """Our special unit
    """
    def __init__(self, in_channels, out_channels, f_size):
        super().__init__()
        self.num_classes = 7
        self.low_k = 5
        self.middle_k = 10
        self.high_k = 20

        self.block_low = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(8, 5), stride=(1, 2), padding=(0, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(128, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(128, 128, kernel_size=(1, 5), stride=(1, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block_middle = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(8, 10), stride=(1, 5), padding=(0, 3), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(128, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )

        self.block_high = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(8, 20), stride=(1, 10), padding=(0, 5), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(128, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )


    def forward(self, x):
        out1 = self.block_low(x)
        #print("out1: ", out1.shape)
        out2 = self.block_middle(x)
        #print("out2: ", out2.shape)
        out3 = self.block_high(x)
        #print("out3: ", out3.shape)
        return torch.cat((out1, out2, out3), 1)


class BlankNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 7
        self.inplanes = 1
        self.layer1 = nn.Sequential(SpecialBlock(1, 128, 5))
        self.conv1 = nn.Conv2d(384, 384, kernel_size=(1,2), stride=(1, 2), bias=False)
        self.conv2 = nn.Conv2d(384, 192, kernel_size=(1,2), stride=(1, 2), bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Sequential(
            nn.Linear(192, self.num_classes)
        )
      
    def forward(self, x):
        out = self.layer1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out
