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


def deepSpeech_thread(audio_que, audio_ent, text_angle_deq, text_deq):
    """Convert speech to text, and classify whether the text is trigger or not

    Args:
		audio_que(que) : que containing audio frames 
        text_angle_deq(deque) : deque containg dictionaries({text:angle}) to send a text and an angle to trigger_detect_thread
        text_deq(deque) : deque to send text to trigger_detect_thread
    """
    
    print("DeepSpeech Thread Join...")
    model = deepspeech.Model('deepSpeech/deepspeech-0.9.3-models.tflite')
    model.enableExternalScorer('deepSpeech/deepspeech-0.9.3-models.scorer')
    stream_context = model.createStream()    
    angles = collections.deque()
    #time.sleep(4)
    audio_ent.wait()
    past_text = " "
    for frame in audio_que.queue[-1].ds_frame:
       
        if frame is not None:
            logging.debug("streaming frame")
            angles.append(audio_que.queue[-1].frame_angle)
            angle = collections.Counter(angles).most_common(1)[0][0]
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            inter_text = stream_context.intermediateDecode()
            if inter_text != "" and inter_text != past_text:
                print(inter_text)
                text_deq.append(inter_text)
                text_angle_deq.append({inter_text:angle})
                past_text = inter_text
        else:
            angle = collections.Counter(angles).most_common(1)[0][0]
            logging.debug("end utterence")
            text = stream_context.finishStream()
            text_deq.append(text)
            text_angle_deq.append({text:angle})
            #print("angles :", angles)
            if angle is None:
                print("Recognized: %s, Angle : None"%(text))
                pass
            else:
                print("Recognized: %s, Angle: %f" %(text, angle))
               
            stream_context = model.createStream()
            angles.clear()
    print("\n dsds end \n")
