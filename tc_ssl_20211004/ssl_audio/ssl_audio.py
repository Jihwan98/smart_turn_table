import numpy as np
import collections
import time
def ssl_audio_thread(ssl_frame_que, ssl_audio_que):
    """
    (320,8),(320,8) is_speech
    if both is_speech -> (640,8) concatenate

    args:
        ssl_frame_que : (idx, i16, is_speech)
        ssl_audio_que : (640, 8)
    """

    maxlen = 2
    ring_buffer = collections.deque(maxlen=maxlen)
    rb_len = 0
    sample = []
    while True:
        ssl_frame = ssl_frame_que.get()
        idx = ssl_frame[0]
        i16 = ssl_frame[1]
        is_speech = ssl_frame[2]

        ring_buffer.append((i16, is_speech))
        rb_len += 1
        if rb_len == maxlen:
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced == maxlen:
                for i, (f, s) in enumerate(ring_buffer):
                    if i == 0:
                        sample = f
                    else:
                        sample = np.concatenate((sample, f), axis=0)
                ssl_audio_que.put((sample, idx))
                time.sleep(0.001)
            ring_buffer.clear()
            rb_len = 0

