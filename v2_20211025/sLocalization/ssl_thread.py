import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, frame_len):
        super().__init__()
        self.num_classes = 14
        self.frame_len = frame_len
    def forward(self, x):
        norm_x = x.clone()

        norm_x = norm_x.view(norm_x.size(0), -1)
        norm_x -= norm_x.min(1, keepdim=True)[0]
        norm_x /= norm_x.max(1, keepdim=True)[0]
        norm_x = norm_x.view(-1, 8, self.frame_len)
        
        #norm_x = (norm_x - min_x[:, None, None]) / (max_x[:, None, None] - min_x[:, None, None])
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
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.block_low = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=self.low_k, padding=1, bias=False),
        )
        self.block_middle = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=self.middle_k, padding=2, bias=False),
        )
        self.block_high = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=self.high_k, padding=3, bias=False),
        )
        self.mp = nn.MaxPool1d(4, stride=2, padding=1)
        
        
        
    def forward(self, x):
        out = self.conv(x)
        out1 = self.block_low(out)
        out1 = torch.cat((out1, out), 1)
        
        out2 = self.block_middle(out)
        out2 = torch.cat((out2, out), 1)
        
        out3 = self.block_high(out)
        out3 = torch.cat((out3, out), 1)
        out =  self.mp(torch.cat((out1, out2, out3), 1))
        return out


class SounDNet(nn.Module):
    def __init__(self, frame_len=2560):
        super().__init__()
        self.num_classes = 7
        self.outc = 32
        self.inplanes = 1
        self.norm = NormBlock(frame_len)
        self.layer1 = nn.Sequential(SpecialBlock(8, self.outc, 5))
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(96*2, self.num_classes*2),
        )

        
    def forward(self, x):
        out = self.norm(x)
        out = self.layer1(out)
        out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out).view(-1, 2, self.num_classes)
        return out[:, 0, :], out[:, 1, :]

def get_angle(predR, predA):
    #print(predR, predA)
    predA = predA.squeeze(0)
    prob = torch.sigmoid(predR)
    temp = (prob > 0.5).sum()
    is_sound = True
    
    label = torch.max(torch.sigmoid(predR), 1)[1].item()
#print(label, temp)
    outputs = (((-1*label) + 7) * 51.4286) - 5.3637 - ((1-torch.sigmoid(predA[label]).item()) * 51.4286)
    if temp > 0:
        return outputs, is_sound
    else:
        is_sound = False
        return outputs, is_sound

def ssl_thread(ssl_queue, angle_list, trig_ent, new_ent):
    """
    at = get_at()
    dict_label = {
        0:at[-1], 1:at[0], 2:at[1], 3:at[2], 4:at[3], 5:at[4], 6:at[5],
        7:at[6], 8:at[7], 9:at[8], 10:at[9], 11:at[10], 12:at[11], 13:at[12]
    }
    """
    net = SounDNet(640)
    net.load_state_dict(torch.load('sLocalization/weight/50.pth', map_location=torch.device('cpu')))
    net.eval()

    
    while True:
        if trig_ent.isSet():
            for i in range(len(ssl_queue.queue)):
                ssl_queue.get()
            trig_ent.clear()
        if new_ent.isSet():
            ssl_queue.get()
            continue
        frame, idx = ssl_queue.get()
        sample = torch.tensor(frame).view(-1,640,8).permute(0, 2, 1).contiguous().float()
        
#if torch.mean(torch.abs(sample)) < 100:
# continue
        out_re, out_an = net(sample)
        angle, is_sound = get_angle(out_re, out_an)
#print(angle)
        # if not is_sound:
            # continue
#pred_angle, pred_idx = get_angle_error(out_an, out_re, dict_label)
        #print(int(pred_angle.detach().numpy()))
        # if ent.isSet():
#angle = int(pred_angle.detach().numpy())
        resi_angle = angle % 10
        if resi_angle >= 6:
            angle = angle + (10-resi_angle)
        else:
            angle = angle - resi_angle
        # print(angle)
        angle_list.append((angle, idx))
        if len(angle_list) > 75:
            angle_list.popleft()

        time.sleep(0.001)
        

if __name__ == '__main__' :
    """
    at = get_at()
    dict_label = {
        0:at[-1], 1:at[0], 2:at[1], 3:at[2], 4:at[3], 5:at[4], 6:at[5],
        7:at[6], 8:at[7], 9:at[8], 10:at[9], 11:at[10], 12:at[11], 13:at[12]
    }
    """
    vad = VADAudio(
            input_rate=16000,
            device=2
    )
    ssl_frames = vad.vad_collector()

    net = SounDNet()
    net.load_state_dict(torch.load('weight/111.pth', map_location=torch.device('cpu')))
    net.eval()        
    go = 0
    for i, (frame,_) in enumerate(ssl_frames):
        if i % 8 == 0:
            pre = frame[1]
        elif (i % 8) > 0 and (i % 8) < 7:
            pre = np.concatenate((pre, frame[1]), 0)
        elif (i % 8) == 7:
            data = np.concatenate((pre, frame[1]), 0)
            t_data = torch.tensor(data).view(-1, 2560, 8).permute(0,2,1).contiguous().float()
            out_re, out_an = net(t_data)
            pred_angle, is_sound = get_angle(out_re, out_an)
            if is_sound:
                print(pred_angle)
            else:
                continue
#time.sleep(1.0)

