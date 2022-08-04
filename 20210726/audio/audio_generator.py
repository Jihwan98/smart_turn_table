"""
@author: Mozilla
    Edited by Kojungbeom
    
reference:
    https://github.com/mozilla/DeepSpeech-examples/blob/r0.9/mic_vad_streaming/mic_vad_streaming.py
"""

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

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 8
    BLOCKS_PER_SECOND = 50  #320 samples

    def __init__(self, callback=None, device=2, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            custom_callback(in_data)
            return (None, pyaudio.paContinue)
        
        # Custom callback
        def custom_callback(in_data):
            """Push raw audio to the buffers
               One for DeepSpeech, the other for SSL
            """
            self.buffer_queue.put(in_data)
            self.ssl_queue.append(in_data)
            
        self.buffer_queue = queue.Queue()
        self.ssl_queue = collections.deque()
        
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def ssl_read(self):
        """Generator for SSL
        """
        while True:
            if self.ssl_queue:
                yield self.ssl_queue.pop()
                self.ssl_queue.clear()
            else:
                time.sleep(0.001)

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection for DeepSpeech."""

    def __init__(self, aggressiveness=2, device=2, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self, task):
        """Generator that yields all audio frames from microphone."""
        while True:
            data = np.frombuffer(self.read(), dtype=np.int16).reshape((-1,8))
            yield data[:, 0].tobytes()


    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None, task='ds'):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator(task=task)
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

