import collections
import numpy as np

def tc_audio_thread(tc_frame_que, tc_audio_que, device_mode):
    """
    is_speech ratio  -> (320, 1) concatenate to  (16000, 1)

    args:
        tc_frame_que : (idx, f32_1c, is_speech)
        tc_audio_que : (16000,1)
    """
    maxlen = 50
    ratio_threashold = 0.5
    ring_buffer = collections.deque(maxlen=maxlen)
    rb_len = 0
    sample = []
    past_mode = 'normal'
    while True:
        tc_frame = tc_frame_que.get()
        idx = tc_frame[0]
        f32_1c = tc_frame[1]
        is_speech = tc_frame[2]
        ring_buffer.append((f32_1c, is_speech))
        rb_len += 1
        dev_mode = device_mode.queue[-1]
        if dev_mode != past_mode:
            ring_buffer.clear()
            rb_len = 0
        past_mode = dev_mode
        if dev_mode == "security" :
            maxlen = 25
            if rb_len == maxlen :
                
                for i, (f, s) in enumerate(ring_buffer):
                    if i == 0:
                        sample = f
                    else :
                        sample = np.concatenate((sample, f), axis = 0)
                tc_audio_que.put((sample, idx))
                ring_buffer.clear()
                rb_len=0
                
        else :
            maxlen = 50
            if rb_len == maxlen:
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio_threashold * maxlen:
                    for i, (f, s) in enumerate(ring_buffer):
                        if i == 0:
                            sample = f
                        else:
                            sample = np.concatenate((sample, f), axis=0)
                    tc_audio_que.put((sample, idx))
                    ring_buffer.clear()
                    rb_len = 0
                else:
                    for i in range(25):
                        ring_buffer.popleft()
                    rb_len = 25
