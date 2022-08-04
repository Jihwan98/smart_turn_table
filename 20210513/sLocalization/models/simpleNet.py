"""
@author: Kojungbeom

"""

import torch
import torch.nn as nn

class SpecialBlock(nn.Module):
    """Our special unit
    """
    def __init__(self, in_channels, out_channels, f_size):
        super().__init__()
        self.num_classes = 7
        self.low_k = 5

        self.block_low = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=(8, self.low_k), stride=(1, int(self.low_k / 2)), padding=(0,round(self.low_k / 4 + 0.5)), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d((1,5))
        )

    def forward(self, x):
        out1 = self.block_low(x)
        return out1


class simpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 8
        self.inplanes = 1
        self.layer1 = nn.Sequential(SpecialBlock(1, 192, 5))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 4))
        self.fc_layer = nn.Sequential(
            nn.Linear(192*4, self.num_classes)
        )
     
    def forward(self, x):
        out = self.layer1(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out

class SpecialBlock2(nn.Module):
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
        '''
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
        '''
        
    def forward(self, x):
        out1 = self.block_low(x)
        #print(out1.shape)
        #ut2 = self.block_middle(x)
        #print(out2.shape)
        #ut3 = self.block_high(x)
        #print(out3.shape)
        return out1
        #eturn torch.cat((out1, out2, out3), 1)

class newNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 7
        self.outc = 64
        self.inplanes = 1
        #self.norm = NormBlock()
        self.layer1 = nn.Sequential(SpecialBlock2(1, self.outc, 5))
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 1))
        self.fc_layer = nn.Sequential(
            nn.Linear(self.outc*4, self.num_classes)
        )
      
    def forward(self, x):
        #out = self.norm(x)
        out = self.layer1(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc_layer(out)
        return out
