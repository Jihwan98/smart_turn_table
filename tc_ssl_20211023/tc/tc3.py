"""Trigger classification
Author: Jungbeom Ko
"""

import os
import pathlib

import numpy as np
#from wakeNet import *
#from audio.audio import * 
import tensorflow as tf
import time

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from firebase.firebase_thread import *

from scipy.io.wavfile import write
from scipy import signal
import python_speech_features
from threading import Thread, Lock, Event

# audio & webcam
import datetime
import json
import cv2
import wave

import pyrebase
import pyaudio
from scipy.io.wavfile import write

# email
import smtplib
from email.mime.text import MIMEText

from scipy.stats import mode
from matrix_lite import gpio 

import cv2
import sys

import shutil


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
pin = 4
min_pulse_ms = 0.5



def calc_mfcc(signal):
    # Load wavefile
    fs = 8000
    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal,
                                            samplerate=8000, 
                                            winlen=0.256,
                                            winstep=0.050,
                                            numcep=16,
                                            nfilt=26,
                                            nfft=2048,
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=False,
                                            winfunc=np.hanning)
    return mfccs.transpose()

def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []
        
    for index, filename in enumerate(in_files):
    
        # Create path from given filename and target item
        path = join(data_path, target_list[int(in_y[index])], 
                    filename)
        
        # Check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue

        # Create MFCCs
        mfccs = calc_mfcc(path)

        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1
            
    return out_x, out_y, prob_cnt

def augmentaton_pitch(file_path):
    speed_rate = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    rate, data = wav.read(file)
    for i, speed in enumerate(speed_rate):
        wav_speed_tune = cv2.resize(np.float32(data), (1, int(len(data) * speed))).squeeze()
        #print('speed rate: %.3f' % speed_rate, '(lower is faster)')
        if len(wav_speed_tune) < 16000:
            pad_len = 16000 - len(wav_speed_tune)
            wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                                  wav_speed_tune,
                                  np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
        else: 
            cut_len = len(wav_speed_tune) - 16000
            wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+16000]
        wav_speed_tune = (wav_speed_tune / np.max(np.abs(wav_speed_tune))*32767).astype('int16')
        path = file[:-4] + '_' + str(i) + '_' +'_speed.wav'
        write(path, 16000, wav_speed_tune)
        
def augmentaton_shift(file_path):
    rate, data = wav.read(file)
    for i in range(4):
        start_ = int(np.random.uniform(-2400,2400))
        if start_ >= 0:
            wav_time_shift = np.int16(np.r_[data[start_:], np.random.uniform(-0.001,0.001, start_)])
        else:
            wav_time_shift = np.int16(np.r_[np.random.uniform(-0.001,0.001, -start_), data[:start_]])
        wav_time_shift = (wav_time_shift / np.max(np.abs(wav_time_shift))*32767).astype('int16')
        path = file[:-4] + '_' + str(i) + '_' +'_shift.wav'
        write(path, 16000, wav_time_shift)

def download_data(data_path, user_email):
    config = {
        "apiKey" : "AIzaSyCsT5QrQpzyQTyaB3Dq7qIDrNMJEoODPpI",
        "authDomain" : "alpha-f18cd.firebaseapp.com",
        "databaseURL" : "https://alpha-f18cd-default-rtdb.firebaseio.com",
        "projectId" : "alpha-f18cd",
        "storageBucket" : "alpha-f18cd.appspot.com",
        "messagingSenderId" : "532001314322",
        "appId" : "1:532001314322:web:a6353cee49e5fa4f7caacb",
        "measurementId" : "G-SHW9Z6D14W",
        "serviceAccount" : "tc/cred/serviceAccountKey.json"
    }
    firebase = pyrebase.initialize_app(config)
    true_dataPath = os.path.join(data_path, 'true')
    temp_filePath = 'temp_files'
    try:
        #os.rmdir(temp_filePath)
        shutil.rmtree(temp_filePath)
    except:
        os.mkdir(temp_filePath)
    try:
        #os.rmdir(data_path)
        shutil.rmtree(data_path)
    except:
        os.mkdir(data_path)
    try:
        #os.rmdir(true_dataPath)
        shutil.rmtree(true_dataPath)
    except:
        os.mkdir(true_dataPath)
    
    root_path = "Voices/" 
    file_path = root_path + user_email
    user_storage = firebase.storage()
    files = user_storage.list_files()
    for i, file in enumerate(files):
        if file_path in file.name:
            print(file.name)
            down_name = temp_filePath + '/' + file.name.split('/')[-1][:-3] + "ogg"
            cvt_name = true_dataPath + '/' + file.name.split('/')[-1][:-3] + "wav"
            user_storage.child(file.name).download(down_name)
            os.system("ffmpeg -i "+ down_name + " -ab 256 -ar 16000 " + cvt_name)

def tc_thread(tc_audio, angle_list, train_ent, device_mode, email_id, loading, rainbow):
    gpio.setFunction(4, 'PWM')
    gpio.setMode(4, 'output')
    sample_rate = 8000
    num_mfcc = 16
    len_mfcc = 16
    resample_size = 8000
    angles = []
    pre_angle = 90
    net = models.load_model('tc/siri.h5')
    user_id = email_id[-1]
    data_path = 'fine_tune'
    true_dataPath = os.path.join(data_path, 'true')
    false_dataPath = os.path.join(data_path, 'false')

    while True:
        if device_mode.queue[-1] == 'training':
            prev_mode = device_mode.queue[-2]
            train_ent.clear()
            download_data(data_path, user_id)
            all_targets = [name for name in os.listdir(data_path) if os.path.isdir(join(data_path, name))]
            true_train_files = []
            # Get list of true case audio files for augmentation
            for name in os.listdir(true_dataPath):
                ture_full_path = true_dataPath + name
                true_train_files.append(true_full_path)
            augmentaton_pitch(true_trian_files)
            augmentaton_shift(true_train_files)
            x_train, y_train = preprocess_data(all_targets, data_path)
            new_model_path = train_new_model(x_train, y_train)
            net = models.load_model(new_model_path)
            device_mode.append(prev_mode)
        frame, idx = tc_audio.get()
        if device_mode.queue[-1]=='security':
            rainbow.put(1)
            fft = np.fft.fft(frame)
            magnitude = np.abs(fft)
            left_spectrum = magnitude[:int(len(magnitude)/2)]
            max_freq = np.argmax(left_spectrum)
            print(max_freq)

            if max_freq > 600 :
                rainbow.put(0)
                start_time = time.time()
                print("abnormal sound detect!")
                time.sleep(0.15)
                angles = []
                for angle, a_idx in angle_list:
                    if (idx - a_idx) < 50: 
                        angles.append(angle)
                mode_angle = mode(angles).mode
                if mode_angle:
                    mode_angle = int(mode_angle[0])
                else:
                    continue
                if mode_angle < 85:
                    mode_angle = 0
                elif mode_angle > 95 and mode_angle < 180:
                    mode_angle = 180
                mode_angle = abs(mode_angle - 180)
                loading.put(1)
                turn_motor(pin, mode_angle, pre_angle, min_pulse_ms,2)
                loading.put(0)
                duration = int(time.time() - start_time)
                pre_angle = mode_angle
                print("mode: ", mode_angle, "angle_idx:", angle_list[-1][1])
            else :
                continue
            
            Video_url = webcam_recording(user_id)
            
            # recording 시간
             
#           CHUNK = 2**10
#           FORMAT = pyaudio.paInt16
#           CHANNELS = 1
            RATE = 16000
            RECORD_SECONDS = 5
 
            # file name
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = now + ".wav"
    
            frame_5 = []
            for i in range(0, duration *4):
                tc_audio.get()
            for i in range(0,10):
                
                frame, idx = tc_audio.get()
                print("음성을 받아서 저장해야됩니다. security")
                frame_5.extend(frame)
                if len(frame_5) == 80000  :
                    print("음성 녹음 직전........!")
                 
                    frame_5 = np.array(frame_5)
                    #print(type(frame_5))
                    #print(frame_5.shape)
                    wf = wave.open(audio_filename, 'wb')
                    write(audio_filename, RATE, frame_5)
                    #wf.setchannels(CHANNELS)
                    #wf.setsampwidth(p.get_sample_size(FORMAT))
                    #wf.setframerate(RATE)
                    #wf.writeframes(b''.join(frame_5))
                    #wf.close()
                    print("record complete!!")
                    config = {    
                        "apiKey": "AIzaSyCsT5QrQpzyQTyaB3Dq7qIDrNMJEoODPpI",
                        "authDomain": "alpha-f18cd.firebaseapp.com",
                        "databaseURL": "https://alpha-f18cd-default-rtdb.firebaseio.com",
                        "projectId": "alpha-f18cd",
                        "storageBucket": "alpha-f18cd.appspot.com",
                        "messagingSenderId": "532001314322",
                        "appId": "1:532001314322:web:a6353cee49e5fa4f7caacb",
                        "measurementId": "G-SHW9Z6D14W",
                        "serviceAccount":"tc/cred/serviceAccountKey.json"
                    }       

                    firebase = pyrebase.initialize_app(config)
                    storage = firebase.storage()

                    path_on_cloud = "Security/Voices/" + user_id  + "/" + audio_filename


                    storage.child(path_on_cloud).put(audio_filename)
                    print("audio firebase upload")
    
                    Audio_url = storage.child(path_on_cloud).get_url(audio_filename)
                    print("audio download")
                    sendMail(user_id, Video_url,Audio_url)

                else :
                    print(len(frame_5))
                    continue
            rainbow.put(1)
                    
                    
        else:
            rainbow.put(1)
            resample = np.array(signal.resample(frame, resample_size), dtype=np.float32)
            sample = np.float32(calc_mfcc(resample))
            sample = sample.reshape(1, 16, 16, 1)
            prediction = net(sample)
            print("idx: ", idx, ", pred: ", prediction)
            if prediction>0.55:
                rainbow.put(0)
                time.sleep(0.15)
                print("trigger!! idx :",idx)
                angles = []
                print("angle_list:", angle_list)
                length = len(angle_list)
                for i in range(length-1, length-26, -1):
                    if (idx - angle_list[i][1]) < 50: 
                        angles.append(angle_list[i][0])
                print("angles:",angles)
                mode_angle = mode(angles).mode
                if mode_angle:
                    mode_angle = int(mode_angle[0])
                else:
                    continue
                print("pre_mode:", mode_angle)
                if mode_angle < 85:
                    mode_angle = 0
                elif mode_angle > 95 and mode_angle < 180:
                    mode_angle = 180
                mode_angle = abs(mode_angle - 180)
                print("after_mode:", mode_angle, 'angle_idx',angle_list[-1][1])
                loading.put(1)
                turn_motor(pin, mode_angle, pre_angle, min_pulse_ms,2)
                loading.put(0)
                rainbow.put(1)
                pre_angle = mode_angle

#print(angle_list)

def delay_time(d):
    return (0.5 * (d-0.5)**2) + 0.025

def turn_motor(pin, angle, pre_angle, min_pulse_ms, step):
    """Function for motor control
    Args:
        pin(int): pin number of matrix voice gpio extension.
        angle(int): current angle
        pre_angle(int): previous angle
        min_pulse_ms(float): min_pulse_ms of servo motor.
        step(int): degree of angular shift per iteration.
        
    """
    step = step if angle > pre_angle else step * (-1)
    dis = abs(pre_angle - angle)

    if dis > 30:
        for i in range(pre_angle, angle+step, step):
            gpio.setServoAngle({
                    'pin': pin,
                    'angle': i,
                    'min_pulse_ms': min_pulse_ms,
            })
            time.sleep(delay_time(abs((i-pre_angle)/dis)))
    else:
        for i in range(pre_angle, angle+step, step):
            gpio.setServoAngle({
                "pin": pin,
                "angle": i,
                "min_pulse_ms": min_pulse_ms,
            })
            time.sleep(0.1)

if __name__ == "__main__":
    sample_rate = 8000
    num_mfcc = 16
    len_mfcc = 16
    resample_size = 8000

    net = models.load_model("eddy_model.h5")

    vad = VADAudio(
            aggressiveness=2,
            input_rate=16000
    )

    frames = vad.vad_collector_TC()
    i = 0
    for frame in frames:
        resample = signal.resample(frame, resample_size)
        resample = np.array(resample, dtype=np.float32)
        sample_ds = np.float32(calc_mfcc(resample))
        sample_ds = sample_ds.reshape(1,
                        16,
                        16,
                        1)
        prediction = net(sample_ds)
        print(prediction)







def webcam_recording(email_id) :
    cap = cv2.VideoCapture(-1)

    if not cap.isOpened() :
        print("Camera open failed!")
        webcam_url = "webcam 없음"  
        
        
    else :
        print("webcam detect")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')        # fourcc : 동영상 압축 코덱을 표현하는 4-문자코드
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        webcam_filename = now + ".avi"
        writer = cv2.VideoWriter(webcam_filename , fourcc, 30.0, (640, 480))

        # cv2.VideoWriter(outputFile, fourcc, frame, size) 
        # : fourcc는 코덱 정보, frame은 초당 저장될 프레임, size는 저장될 사이즈


        # recording
        start_time = time.time()
        while True :
            ret, img_color = cap.read()
            if ret == False :
                continue
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("Color", img_color)

            writer.write(img_color)

            if time.time() - start_time > 5 :
                break

        cap.release()
        writer.release()
        #cv2.destroyAllWindows()
        print("release")
        # upload
        config = {
            "apiKey": "AIzaSyCsT5QrQpzyQTyaB3Dq7qIDrNMJEoODPpI",
            "authDomain": "alpha-f18cd.firebaseapp.com",
            "databaseURL": "https://alpha-f18cd-default-rtdb.firebaseio.com",
            "projectId": "alpha-f18cd",
            "storageBucket": "alpha-f18cd.appspot.com",
            "messagingSenderId": "532001314322",
            "appId": "1:532001314322:web:a6353cee49e5fa4f7caacb",
            "measurementId": "G-SHW9Z6D14W",
            "serviceAccount": "tc/cred/serviceAccountKey.json"
        }
        firebase = pyrebase.initialize_app(config)
        storage = firebase.storage()
    

        path_on_cloud = "Security/Videos/" + email_id + "/" + webcam_filename
        storage.child(path_on_cloud).put(webcam_filename)
        print("webcam firebase upload")

        webcam_url = storage.child(path_on_cloud).get_url(webcam_filename)
        print("webcam download")

    return webcam_url


"""
def audio_recording (email_id) :
    # upload
    config = {
        "apiKey": "AIzaSyCsT5QrQpzyQTyaB3Dq7qIDrNMJEoODPpI",
        "authDomain": "alpha-f18cd.firebaseapp.com",
        "databaseURL": "https://alpha-f18cd-default-rtdb.firebaseio.com",
        "projectId": "alpha-f18cd",
        "storageBucket": "alpha-f18cd.appspot.com",
        "messagingSenderId": "532001314322",
        "appId": "1:532001314322:web:a6353cee49e5fa4f7caacb",
        "measurementId": "G-SHW9Z6D14W"
    }
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    path_on_cloud = "Audios/" + email_id + "/" + filename
    storage.child(path_on_cloud).put(filename)
    print("upload")
    Audio_url = storage.child(path_on_cloud).get_url(filename)
    print("download")
    
    return Audio_url
"""




def sendMail(email_id,Video_url, Audio_url) :
    # Send email
    
    content = "Video : ", Video_url, "\n", "Audio : ", Audio_url
    content = ''.join(content)
    #print(content)
    email_from = "lge000325@gmail.com"
    email_to = email_id
    email_subject = "Security 알림!"
    #email_content = "당신의 집에서 이상한 지되었습니다"
    email_content = content

    msg = MIMEText(email_content)
    msg['From'] = email_from
    msg['To'] = email_to
    msg['Subject'] = email_subject

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login("lge000325@gmail.com", "yocloqojcsbcptgg")       # 보내는 자의 이메일과 비번
    smtp.sendmail("lge000325@gmail.com", "lge000325@gmail.com", msg.as_string())     # 받는 자의 이메일 email_id
    print(msg.as_string())
    print("send email!")
    smtp.quit()
