import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
from audio.audio_generator import *


class AudioFrame:
    """This class has audio frame's informations. It act like a structure.
	
    member variable:
        frame_num : use as frame_idx, it reset over 500
	
    instance variable:
        frame_idx : index, 0~500 (auto set)
        ds_frames(Audio) : generator which generate raw audio data 
        ssl_frames(Audio) : generator which generate multi-channel raw audio data 
        frame_time : time when object created (auto set)
        angle : angle
    """
    frame_num = 0

    def __init__(self, ds_frame = None, ssl_frame = None, angle = None):
        self.frame_idx = AudioFrame.frame_num
        self.ds_frame = ds_frame
        self.ssl_frame = ssl_frame
        self.frame_time = time.time()
        self.frame_angle = angle
        AudioFrame.frame_num += 1
        if AudioFrame.frame_num > 500:
            AudioFrame.frame_num = 0

    def ShowAll(self):
        print("frame_idx :", self.frame_idx)
        print("ds_frame :", self.ds_frame)
        print("ssl_frame :", self.ssl_frame)
        print("frame_time :", self.frame_time)
        print("angle :", self.frame_angle)


def audio_thread(audio_que, audio_ent):
    """Get audio frames and make an audio deque

    Args:
        audio_que(queue) : queue to send audio frame
    
    """

    vad = VADAudio(
            aggressiveness=2,
            input_rate=16000
            )

    audio_que.put(AudioFrame(vad.vad_collector(), vad.ssl_read()))
    audio_ent.set()
    if len(audio_que.queue) > 500:
        audio_que.get()
