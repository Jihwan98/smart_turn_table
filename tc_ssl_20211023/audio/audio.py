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

    FORMAT = pyaudio.paFloat32
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 8
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
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
            self.buffer_queue.put((in_data, self.idx))
            if self.idx > 4999:
               self.idx = 0
            self.idx += 1
        
        self.idx = 0
        self.buffer_queue = queue.Queue()
        
        self.device = 2
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
        data, idx = self.buffer_queue.get()
        return data, idx

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection for DeepSpeech."""

    def __init__(self, aggressiveness=2, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        a = (2**15)-1
        b = 2**15
        while True:
            _data, _idx = self.read()
            data_f32 = np.frombuffer(_data, dtype=np.float32).reshape((-1,8))
            data_f32_1c = data_f32[:,0]
            data_i16 = np.int16(((data_f32 + 1.0) /2)* a - b)
            data_i16_1c = data_i16[:,0].tobytes()
            yield _idx, data_f32, data_f32_1c, data_i16, data_i16_1c


    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        triggered = False

        for idx, f32, f32_1c, i16, i16_1c in frames:
            if len(i16_1c) < 640:
                return

            is_speech = self.vad.is_speech(i16_1c, self.sample_rate)
            # ssl_frame = (idx, i16, is_speech)
            # tc_frame = (idx, f32_1c, is_speech)
            all_frame = (idx, f32_1c, i16, is_speech)
            yield all_frame

def audio_thread(audio2SSL_audio, audio2TC_audio, loading, rainbow):
    print('Audio Thread Join...')
    vad = VADAudio(
            aggressiveness=2,
            input_rate=16000
    )
    rainbow.put(1)
    for frame in vad.vad_collector():
        if loading.queue[-1] == 1:
            pass
        else:
            # audio2SSL_audio.put(ssl)
            # audio2TC_audio.put(tc)
            audio2TC_audio.put(frame)
            time.sleep(0.001)

if __name__ == "__main__":
    vad = VADAudio(
            aggressiveness=2,
            input_rate=16000
    )
    
    for ssl, tc in vad.vad_collector():
        # print("ssl :", ssl)
        # print("tc :", tc)
        print("ssl.shape : ", ssl[1].shape)
        print("tc.shape : ", tc[1].shape)

