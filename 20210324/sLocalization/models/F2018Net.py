"""
@author: Kojungbeom

"""

import torch
import torch.nn as nn

class F2018Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 7
        self.inplanes = 8
        self.block = nn.Sequential(
            nn.Conv1d(self.inplanes, 96, padding=3, kernel_size=7, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(96, 96, padding=3, kernel_size=7, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 128, padding=2, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, padding=2, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*20, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_classes)
        )

    def forward(self, x):
        out = self.block(x)
        return out
