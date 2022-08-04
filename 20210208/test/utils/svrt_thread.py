# 쓰레드를 사용하기 위해서는 threading 모듈을 import해야 한다.
# queue를 사용하기 위해서 queue를 import해야 한다.
# time 모듈은 Thread 상에서 처리를 잠시 멈추게 하는 기능이 있다.
# Install Firebase Admin SDK at python : pip install --upgrade firebase-admin
# To initialize Firebase Admin SDK, you have to import firebase_admin, credentials and firestore 

from threading import Thread
from threading import Lock
from queue import Queue
import socket
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import usb.core
import os

from ssl.models.simpleNet import simpleNet
from matrix_lite import gpio
import torch
import numpy as np
import argparse
import sys
from utils import *

import logging
import threading, collections, queue, os, os.path
import deepspeech
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal


FORMAT = pyaudio.paInt16
CHANNELS = 8
RATE = 96000
energy_array = np.array([0] * 7)
DECREMENT = 5
INCREMENT = 20
MAX_VALUE = 200
RECORD_SECONDS = 5
led_dict = {0:0, 1:2, 2:5, 3:8, 4:10, 5:13, 6:15}
angle_dict = {0:160, 1:120, 2:60, 3:20}
gpio.setFunction(2, 'PWM')
gpio.setMode(2, 'output')
angle_list = [0,1,2,3]

def send_thread(sock, q, lock) :
    # lock을 acquire하면 해당 쓰레드만 공유 데이터에 접근
    # lock을 release 해야만 다른 쓰레드에서 공유 데이터에 접근
    """
    Monitor queue for new messages, send them to client as
    they arrive

    """
    while True:
        lock.acquire() # 스레드 동기화를 막기위한 락
        if q.empty():
            lock.release() # 원래 lock을 획득한 프로세스나 스레드뿐만 아니라 모든 프로세스나 스레드에서 호출가능
            time.sleep(0.1)
        else:
            lock.release() # 업데이트 후 락 해제
            sendData = q.get()
            #sock.send(sendData.encode()) # Client -> Server 데이터 송신
            print(sendData)
            time.sleep(0.1)

def recv_thread(sock, q, lock) :
    """
    Receive messages from client and broadcast them to
    other clients until client disconnects

    """
    while True:
        recvData = sock.recv(1024)
        print('client :', recvData.decode()) # Server -> Client 데이터 수신
        lock.acquire() # lock이 설정된 이상 다음 이 lock를 호출할 때 thread는 대기를 한다.
        q.put(recvData)
        lock.release()
        time.sleep(0.1)

def get_doc_ref(email_id):
    """
    initialize Firebase Admin SDK
    set return value as user's document root
    """
    cred = credentials.Certificate("./json/alpha.json")
    firebase_admin.initialize_app(cred)

    db = firestore.client()

    return db.collection(u'setting').document(email_id)

def on_snapshot(doc_snapshot, changes, read_time):
    """
    Create a callback on_snapshot function to capture changes
    """
    for doc in doc_snapshot:
        cmd_txt = doc.to_dict()["Command_text"]
        print(f'Command changed : {cmd_txt}')


def get_command(changed, email_id) :
    while True:
        if changed[0] == [[1]]:
            doc_ref = get_doc_ref(email_id)
            break
        else:
            time.sleep(0.1)
        
    
    while True:
        if changed[0] == [[1]]:
            #doc_ref = get_doc_ref(email_id)
            doc_watch = doc_ref.on_snapshot(on_snapshot)
            break
            
        else:
            time.sleep(1.0)
            #None


def usb_detection_thread(changed) :
    """
    This will mount the drive so that the ordinary Pi user can write to it
    connected usb의 정보를 얻어 mount 명령어로 (wifi-id, pw, email, command)txt를 만든다
    usb를 통해 주어진 wifi으로 연결되도록 한다
    """
    conf = {}
    while True:
        #print(changed[0])
        dev = usb.core.find(idVendor=0x0781, idProduct=0x5590)
        if dev and not changed[0]==[1]:
            # mount sh
            print("Start Mount")
            time.sleep(5)
            os.system("./scripts/usb_mount.sh")
            time.sleep(5)
            with open('/home/pi/usb/config.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split('=')
                    if line[0] == "wifi_id":
                        conf["wifi_id"] = line[1][:-1]
                    elif line[0] == "wifi_pw":
                        conf["wifi_pw"] = line[1][:-1]
                    elif line[0] == "email":
                        conf["email"] = line[1][:-1]
                    elif line[0] == "command":
                        conf["command"] = line[1][:-1]
            with open('wpa_supplicant.conf', 'w') as f:
                f.write("ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n")
                f.write("update_config=1\n")
                f.write("country=KR\n")
                f.write("network={\n")
                f.write('\tssid="' +  conf["wifi_id"] + '"\n')
                f.write('\tpsk="' + conf["wifi_pw"] + '"\n')
                f.write("}")
            os.system("./scripts/wifi.sh")
            time.sleep(15)
            print("Network Connected")
            changed[0] = [1]
        else:
            time.sleep(1)
            
def decrease_pots():
    for i in range(len(energy_array)):
        energy_array[i] -= DECREMENT
        if energy_array[i] < 0:
            energy_array[i] = 0
            
def increase_pots(y=7):
    everloop = ['black'] * led.length
    if y == 7:
        led.set(everloop)
        pass
    else:
        energy_array[y] += INCREMENT
        if energy_array[y] > MAX_VALUE:
            energy_array[y] = MAX_VALUE
        if np.max(energy_array) > 20:
            everloop[led_dict[np.argmax(energy_array)]] = {'b':50}
            print(np.argmax(energy_array))
            led.set(everloop)
        else:
            led.set(everloop)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 8
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            my_callback(in_data)
            return (None, pyaudio.paContinue)
        def my_callback(in_data):
            self.buffer_queue.put(in_data)
            self.ssl_queue.append(in_data)
        #if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
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
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()


    def ssl_read(self):
        while True:
            if self.ssl_queue:
                yield self.ssl_queue.pop()
                self.ssl_queue.clear()
            else:
                pass
                time.sleep(0.001)

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=2, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self, task):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            if task == 'ds':
                while True:
                    data = np.frombuffer(self.read(), dtype=np.int16).reshape((-1,8))
                    yield data[:, 0].tobytes()
        else:
            while True:
                yield self.read_resampled()

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


def deep(frames):
    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    #print('Initializing model...')
    #logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model)
    if ARGS.scorer:
        #logging.info("ARGS.scorer: %s", ARGS.scorer)
        model.enableExternalScorer(ARGS.scorer)
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            if ARGS.savewav: wav_data.extend(frame)
        else:
            #sendData = str("Detection")
            #clientSock.send(sendData.encode())
            if spinner: spinner.stop()
            logging.debug("end utterence")
            if ARGS.savewav:
                vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                wav_data = bytearray()
            text = stream_context.finishStream()
            print("Recognized: %s" % text)
            if ('f' in text) and ('r' in text) :
                tc = str("Trigger")
                #clientSock.send(tc.encode())
                #recvData = clientSock.recv(1024)
            
            stream_context = model.createStream()

def sssl(frames):
    net = simpleNet()
    net.load_state_dict(torch.load('weight/115.pth', map_location=torch.device('cpu')))
    net.eval()        
    audio = Audio()
    frames = audio.read()
    for frame in frames:
        t_data = torch.tensor(np.frombuffer(frame, dtype=np.int16))
        t_data = t_data.view(-1,8).view(-1,320,1,8).permute(0,2,3,1).contiguous()
        t_data = t_data.float()
        if torch.mean(torch.abs(t_data)) < 100:
            increase_pots()
            decrease_pots()
            continue
        outputs = net(t_data)
        _, pred = outputs.max(1)
        increase_pots(int(pred))
        decrease_pots()
        max_in = np.argmax(energy_array)
        '''
        if max_in in angle_list:
            gpio.setServoAngle({
                "pin": 2,
                "angle": angle_dict[max_in],
                "min_pulse_ms":0.5,
            })
        '''
        time.sleep(0.001)


def error_handling(problem):
    """
    사용자가 직접 에러를 발생시키는 기능
    """
    raise Exception(problem, "Erorr")
