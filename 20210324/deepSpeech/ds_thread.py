"""
@author: Mozilla
    Edited by Kojungbeom

reference:
    https://github.com/mozilla/DeepSpeech-examples/blob/r0.9/mic_vad_streaming/mic_vad_streaming.py
"""

from matrix_lite import gpio
import os
import time, logging
import threading
import collections
import queue
import deepspeech
import numpy as np
import chars2vec

from ctypes import *
from scipy import signal
from trigger.trigger_thread import *

pin = 4
min_pulse_ms = 0.5

def deepSpeech_thread(frames, tods_deq, angle_deq, motor_ent):
    # Matrix Voice GPIO Setup
    gpio.setFunction(4, 'PWM')
    gpio.setMode(4, 'output')

    model = deepspeech.Model('deepSpeech/deepspeech-0.9.3-models.tflite')
    c2v_model = chars2vec.load_model('eng_50')
    trigger_model = get_updated_model('friend', c2v_model, 'trigger/data/others.txt')
    #model.enableExternalScorer('deepSpeech/deepspeech-0.9.3-models.scorer')
    angle = 0
    pre_angle = 0
    stream_context = model.createStream()
    for frame in frames:
        if tods_deq:
            print("New Trigger Detected")
            new_trigger = tods_deq.popleft()
            new_trigger = new_trigger.replace(" ", "").lower()
            trigger_model = get_updated_model(new_trigger, c2v_model, 'trigger/data/others.txt')
            print("Finish Train")
        
        if frame is not None:
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            angle = angle_deq[-1]
        else:
            logging.debug("end utterence")
            text = stream_context.finishStream()
            print("Recognized: %s, Angle: %f" %(text, angle))

            preds = []
            for t in text.split(' '):
                if "" == t:
                    continue
                if "'" in t:
                    continue
                #in_t = indexing(t)
                in_t = list([t])
                print(in_t)
                in_t = c2v_model.vectorize_words(in_t)
                print(2)
                pred = trigger_model.predict(in_t)
                print(3)
                preds.append(pred)
                
            if 1 in preds:
                print("Trigger!!", pin)
                motor_ent.clear()
                turn_motor(pin, angle, pre_angle, min_pulse_ms, 2)
                motor_ent.set()
                pre_angle = angle

            stream_context = model.createStream()

def turn_motor(pin, angle, pre_angle, min_pulse_ms, step):
    step = step if angle > pre_angle else step * (-1)
    for i in range(pre_angle, angle, step):
        gpio.setServoAngle({
            "pin": pin,
            "angle": i,
            "min_pulse_ms": min_pulse_ms,
        })
        time.sleep(0.015)

    gpio.setServoAngle({
        "pin": pin,
        "angle": angle,
        "min_pulse_ms": min_pulse_ms,
    })

