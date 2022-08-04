
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
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
    RATE_PROCESS_SSL = 16000
    RATE_PROCESS_TC = 8000
    RATE_PROCESS = 16000
    CHANNELS = 8
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            custom_callback(in_data)
            return (None, pyaudio.paContinue)

        def custom_callback(in_data):
            """Push raw audio to the buffers
               One for TC, the other for SSL
            """
            self.idx += 1
            self.buffer_queue.put((in_data, self.idx))
            self.ssl_queue.put((in_data, self.idx))
        #if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.idx = 0
        self.buffer_queue = queue.Queue()
        self.ssl_queue = queue.Queue()

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

    def read_TC(self):
        """Return a block of audio data, blocking if necessary."""
        data, idx = self.buffer_queue.get()
        return (data,idx)
    def read_SSL(self):
        """Return a block of audio data, blocking if necessary."""
        data, idx = self.ssl_queue.get()
        return (data,idx)

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.float32)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.float32)
        return resample16.tostring()

    def resample_8000_1(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data32 = np.fromstring(string=data, dtype=np.float32)
        resample_size = int(len(data32) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data32, resample_size)
        resample32 = np.array(resample, dtype=np.float32)[: 0]
        return resample32.tostring()

    def resample_all(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data32 = np.fromstring(string=data, dtype=np.float32)
        resample_size = int(len(data32) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data32, resample_size)
        resample32 = np.array(resample, dtype=np.float32)[: 0]
        return resample32.tostring()

    def resample_16000_8(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data32 = np.fromstring(string=data, dtype=np.float32)
        resample_size = int(len(data32) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data32, resample_size)
        resample32 = np.array(resample, dtype=np.float32)
        return resample32.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read_16000_8_SSL(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        data, idx = self.ssl_queue.get()
        return self.resample_8000_1(data=data, input_rate=16000), idx

    def read_8000_1_TC(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        data, idx = self.buffer_queue.get()
        return self.resample_8000_1(data=data, input_rate=16000), idx

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)



class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=2, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator_TC(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS_TC:
            while True:
                a, b = self.read()
                yield (a,b)
        else:
            while True:
                #print("here")
                
                yield self.read_8000_1_TC()

    def frame_generator_SSL(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS_SSL:
            while True:
                yield self.read_SSL()
        else:
            while True:
                #print("here")
                yield self.read_16000_8_SSL()

    def vad_collector_TC(self, ratio_threshold=0.5, step=8000, maxlen=50, frames=None):
        if frames is None: frames = self.frame_generator_TC()
        ring_buffer = collections.deque(maxlen=maxlen)
        rb_len = 0	# Length of the ring_buffer
        sample = []		#
        # variables for converting data type
        a = (2**15)-1
        b = 2**15
        for frame, idx  in frames:
            # audio data which has float32 data type
            data_f32 = np.frombuffer(frame, dtype=np.float32).reshape(-1)
            # Convert data type float32 to int16 
            data_i16 = np.int16(((data_f32 + 1.0) / 2)* a - b).tobytes()
            # VAD needs int16 data
            is_speech = self.vad.is_speech(data_i16, self.RATE_PROCESS)
            # TC(Trigger classification) model needs float32 data
            ring_buffer.append((data_f32, is_speech))
            rb_len += 1
            if rb_len == maxlen:
                num_voiced = len([f for f, speech in ring_buffer if speech])
                print("num: ", num_voiced, "len: ", len(ring_buffer))
                if num_voiced > ratio_threshold * maxlen:
                    for idx, (f, s) in enumerate(ring_buffer):
                        data = np.frombuffer(f, dtype=np.float32).reshape(-1)
                        if idx == 0:
                            sample = data
                        else:
                            sample = np.concatenate((sample, data), axis=0)
                    ring_buffer.clear()
                    rb_len = 0
                    yield sample
                else:
                    for i in range(25):
                        ring_buffer.popleft()
                    rb_len = 25

    def vad_collector_SSL(self, ratio_threshold=0.5, step=8000, maxlen=2, frames=None):
        if frames is None: frames = self.frame_generator_SSL()
        ring_buffer = collections.deque(maxlen=maxlen)
        rb_len = 0
        sample = []		#
        # variables for converting data type
        a = (2**15)-1
        b = 2**15
        for frame, idx in frames:
            # audio data which has float32 data type
            data_f32 = np.frombuffer(frame, dtype=np.float32).reshape(-1, 8)
            # Convert data type float32 to int16 
            data_i16 = np.int16(((data_f32 + 1.0) / 2)* a - b)
            is_speech = self.vad.is_speech(data_i16[:, 0].tobytes(), self.RATE_PROCESS)
            # SSL needs int16 data
            ring_buffer.append((data_i16, is_speech))
            rb_len += 1
            if rb_len == maxlen:
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced == maxlen:
                    for idx, (f, s) in enumerate(ring_buffer):
                        data = np.frombuffer(f, dtype=np.int16).reshape(-1, 8)
                        if idx == 0:
                            sample = data
                        else:
                            sample = np.concatenate((sample, data), axis=0)
                    ring_buffer.clear()
                    rb_len = 0
                    yield sample, idx
                else:
                    ring_buffer.popleft()
                    rb_len = 1

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None, task='ds'):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator_TC()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        rb_len = 0
        sample = []		#
        triggered = False
        # variables for converting data type
        a = (2**15)-1
        b = 2**15
        for frame, bb in frames:
            if len(frame) < 640:
                return

            # audio data which has float32 data type
            data_f32 = np.frombuffer(frame, dtype=np.float32).reshape(-1)
            # Convert data type float32 to int16 
            data_i16 = np.int16(((data_f32 + 1.0) / 2)* a - b).tobytes()
            is_speech = self.vad.is_speech(data_i16[:, 0], self.RATE_PROCESS)
            # SSL needs int16 data
            if not triggered:
                ring_buffer.append((data_i16, is_speech, bb))
                #ring_buffer.append((frame, is_speech, bb))
                num_voiced = len([f for f, speech, _ in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s, bb in ring_buffer:
                        yield f, bb 
                    ring_buffer.clear()

            else:
                yield frame, bb
                ring_buffer.append((frame, is_speech, bb))
                num_unvoiced = len([f for f, speech, _ in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None, bb
                    ring_buffer.clear()


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

