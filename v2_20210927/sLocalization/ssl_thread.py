import torch
import torch.nn as nn


import time, logging
import threading
import collections
import queue
import os, os.path
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
from audio import *


class NormBlock(nn.Module):
    """Our special unit
    """
    def __init__(self):
        super().__init__()
        self.num_classes = 14
    def forward(self, x):
        norm_x = x.clone()
        max_x = torch.max(torch.max(x, 2)[0], 1)[0]
        min_x = torch.min(torch.min(x, 2)[0], 1)[0]
        norm_x = (norm_x - min_x[:, None, None]) / (max_x[:, None, None] - min_x[:, None, None])
        return norm_x

class SpecialBlock(nn.Module):
    """Our special unit
    """
    def __init__(self, in_channels, out_channels, f_size):
        super().__init__()
        self.num_classes = 7
        self.low_k = 3
        self.middle_k = 5
        self.high_k = 7
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block_low = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=self.low_k, padding=1),
        )
        self.mp1 = nn.Sequential(
            nn.MaxPool1d(4)
        )
        self.block_middle = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=self.middle_k, padding=2),
        )
        self.mp2 = nn.Sequential(
            nn.MaxPool1d(4)
        )
        self.block_high = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=self.high_k, padding=3),
        )
        self.mp3 = nn.Sequential(
            nn.MaxPool1d(4)
        )
        
        #self.mp1 = nn.MaxPool1d(4)
    def forward(self, x):
        out = self.conv(x)
        
        out1 = self.block_low(out)
        out1 = torch.cat((out1, out), 1)
        
        out2 = self.block_middle(out)
        out2 = torch.cat((out2, out), 1)
        
        out3 = self.block_high(out)
        out3 = torch.cat((out3, out), 1)
        
        out =  torch.cat((self.mp1(out1), self.mp2(out2), self.mp3(out3)), 1)
        
        #return out1
        return out


class SounDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 14
        self.outc = 32
        self.inplanes = 1
        self.norm = NormBlock()
        self.layer1 = nn.Sequential(SpecialBlock(8, self.outc, 5))

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(96*2, self.num_classes*2),
        )

        
    def forward(self, x):
        out = self.norm(x)
        out = self.layer1(out)
        out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out).view(-1, 2, 14)
        return out[:, 0, :], out[:, 1, :]


def get_at():
    angle_table_base = np.array([
        174.6362618184203,
        123.21703953759831,
        71.78740177723832,
        20.351193195418745,
        -31.071493812846292,
        -82.49828732852542,
        -133.93144915979693]
    ) + 180

    at_a = angle_table_base - 12.86
    at_b = (angle_table_base + 12.86) % 360

    at = np.sort(np.append(at_a, at_b))
    return at

      
def get_angle_error(pred_angle, pred_label, dict_label):
    #print("pa:", pred_angle.shape)
    #print("pl:", pred_label.shape)
    p_idx = int(torch.argmax(pred_label[0]))
    p_angle = dict_label[p_idx] + pred_angle[0][p_idx] * 25.71
    return p_angle, p_idx

def ssl_thread(ssl_queue, angle_list, train_ent):
    at = get_at()
    dict_label = {
        0:at[-1], 1:at[0], 2:at[1], 3:at[2], 4:at[3], 5:at[4], 6:at[5],
        7:at[6], 8:at[7], 9:at[8], 10:at[9], 11:at[10], 12:at[11], 13:at[12]
    }
    net = SounDNet()
    net.load_state_dict(torch.load('sLocalization/weight/476.pth', map_location=torch.device('cpu')))
    net.eval()

    while True:
        if train_ent.isSet() == False:
            train_ent.wait()
        frame, idx = ssl_queue.get()
        sample = torch.tensor(frame).view(-1,640,8).permute(0, 2, 1).contiguous().float()
        if torch.mean(torch.abs(sample)) < 100:
            continue
        out_re, out_an = net(sample) 
        pred_angle, pred_idx = get_angle_error(out_an, out_re, dict_label)
        #print(int(pred_angle.detach().numpy()))
        # if ent.isSet():
        angle = int(pred_angle.detach().numpy())
        resi_angle = angle % 5
        if resi_angle >= 3:
            angle = angle + (5-resi_angle)
        else:
            angle = angle - resi_angle
        #print(angle)
        angle_list.append((angle, idx))
        
        if len(angle_list) > 75:
            angle_list.popleft()


if __name__ == '__main__' :
    at = get_at()
    dict_label = {
        0:at[-1], 1:at[0], 2:at[1], 3:at[2], 4:at[3], 5:at[4], 6:at[5],
        7:at[6], 8:at[7], 9:at[8], 10:at[9], 11:at[10], 12:at[11], 13:at[12]
    }
    vad = Audio(
            input_rate=16000,
            device=2
    )
    ssl_frames = vad.ssl_read()

    net = SounDNet()
    net.load_state_dict(torch.load('sLocalization/weight/484.pth', map_location=torch.device('cpu')))
    net.eval()        

    for frame in ssl_frames:
        t_data = torch.tensor(frame).view(-1,640,8).permute(0, 2, 1).contiguous().float()
        out_re, out_an = net(t_data)
        pred_angle, pred_idx = get_angle_error(out_an, out_re, dict_label)
        print(pred_angle)
        time.sleep(0.001)

