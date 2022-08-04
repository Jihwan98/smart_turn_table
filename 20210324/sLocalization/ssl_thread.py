"""
@author: Kojungbeom
    
reference:
    https://github.com/matrix-io/odas/blob/master/demo/matrix-demos/matrix-odas.cpp
"""

import sys
import os
sys.path.append('sLocalization/models/')

from simpleNet import simpleNet, newNet
from blankNet import BlankNet
from matrix_lite import led
import time
import torch
import numpy as np
import collections
import torch.nn.functional as F

values = np.array([0] * 7)
DECREMENT = 5
INCREMENT = 20
MAX_VALUE = 200
THRESHOLD = 20
led_dict = {0:0, 1:2, 2:5, 3:8, 4:10, 5:13, 6:15}

def increase_pots(predict=None):
    everloop = ['black'] * led.length
    if predict:
        values[predict] += INCREMENT
        if values[predict] > MAX_VALUE:
            values[predict] = MAX_VALUE
        if np.max(values) > THRESHOLD:
            everloop[led_dict[np.argmax(values)]] = {'b':50}
            led.set(everloop)
        else:
            led.set(everloop)
    else:
        led.set(everloop)


def decrease_pots():
    for i in range(len(values)):
        values[i] -= DECREMENT
        if values[i] < 0:
            values[i] = 0


angles = {0:90, 1:39, 2:-12, 3:-12, 4:192, 5:192, 6:141}

def sLocalizer(frames, angle_deq, motor_ent):
    angle_deq.append(0)
    #net = newNet()
    net = BlankNet()
    net.load_state_dict(torch.load('sLocalization/weight/197.pth', map_location=torch.device('cpu')))
    net.eval()        
    for frame in frames:
        t_data = torch.tensor(np.frombuffer(frame, dtype=np.int16))
        t_data = t_data.view(-1,8).view(-1,1,320,8)
        t_data = t_data.float()

        angle = 0 
        if torch.mean(torch.abs(t_data)) < 200:
            increase_pots()
            decrease_pots()
            continue
        
        outputs = net(t_data)
        c_outputs = outputs.clone()
        _, pred = outputs.max(1)
        _, top1 = outputs.topk(1)
        top1_index = int(top1[0][0])

        if top1_index == 3 or top1_index == 4:
            continue
        all_psb = F.softmax(c_outputs, dim=1)[0]

        top1_psb = all_psb[(top1_index)]
        top1_left = all_psb[(top1_index-1)%7]
        top1_right = all_psb[(top1_index+1)%7]

        top1_near, top1_near_index = \
            (top1_left, (top1_index-1)%7) if top1_left > top1_right else (top1_right, (top1_index+1)%7)
               

        ratio = [(top1_psb / (top1_psb + top1_near)),
                (top1_near / (top1_psb + top1_near))]

        angle += angles[top1_index] * ratio[0]
        angle += angles[top1_near_index] * ratio[1]
        
        if angle > 180:
            angle = 180
        if angle < 0:
            angle = 0

        increase_pots(predict=int(pred))
        decrease_pots()
        max_index = np.argmax(values)

        if top1_index == max_index:
            if len(angle_deq) > 10:
                angle_deq.popleft()
                angle_deq.append(int(angle))
            else:
                angle_deq.append(int(angle))
        else:
            time.sleep(0.001)
                
            


