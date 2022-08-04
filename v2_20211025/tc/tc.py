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
from scipy.io import wavfile as wav
from scipy import signal
import python_speech_features
from threading import Thread, Lock, Event

# audio & webcam
import datetime
import json
import cv2
import wave

# audio 이긴 한데.. 일시적으로 만든거라 matrix voice 쓰면 필요없어짐
import pyrebase
import pyaudio
from scipy.io.wavfile import write

# email
import smtplib
from email.mime.text import MIMEText

from scipy.stats import mode
from matrix_lite import gpio 

import sys
from tflite_runtime.interpreter import Interpreter
import librosa

from os.path import isdir, join
from os import listdir
import random


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
pin = 4
min_pulse_ms = 0.5
sampling_rate = 8000
fs = 8000

def calc_mfcc(signal):
    # Load wavefile
    fs = 8000
    """
    if signal.shape[0] > 8000:
        signal = signal[:8000]
    elif signal.shape[0] < 8000:
        signal = np.concatenate((signal, np.zeros(8000-len(signal))), 0)
    """
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

dataset_path = 'data'
all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
target_list = all_targets

def calc_mfcc_train(path):

    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)
    #print(fs)
    if signal.shape[0] > 8000:
        signal = signal[:8000]
    elif signal.shape[0] < 8000:
        signal = np.concatenate((signal, np.zeros(8000-len(signal))), 0)
    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal,
                                            samplerate=fs,
                                            winlen=0.256,
                                            winstep=0.050,
                                            numcep=num_mfcc,
                                            nfilt=26,
                                            nfft=2048,
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=False,
                                            winfunc=np.hanning)
    return mfccs.transpose()

#dataset_path = 'data'

def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []

    for index, filename in enumerate(in_files):

        # Create path from given filename and target item
        path = join(dataset_path, target_list[int(in_y[index])],
                    filename)

        # Check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue

        # Create MFCCs
        mfccs = calc_mfcc_train(path)

        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1

    return out_x, out_y, prob_cnt


def tc_thread(tc_audio, angle_list, device_mode,email_id,loading,rainbow, trig_ent, new_ent):
    gpio.setFunction(4, 'PWM')
    gpio.setMode(4, 'output')
    sample_rate = 8000
    num_mfcc = 16
    len_mfcc = 16
    resample_size = 8000
    angles = []
    pre_angle = 90
    it = 0
    #net = models.load_model('tc/v0.1_Eddy_TC_20211002.h5')
    
    interpreter = Interpreter('tc/v0.1_Eddy_TC_20211002.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    while True:
        user_id = email_id[-1]
        frame, idx = tc_audio.get()

        if device_mode.queue[-1] == 'training':
            new_ent.set()
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

            root_path = "Voices/" 
            user_email = "test@test.com/eddy"
            file_path = root_path + user_email
            user_storage = firebase.storage()
            files = user_storage.list_files()
            os.mkdir("temp_data")
            os.system("rm -r data/true")
            os.mkdir("data/true")
            for i, file in enumerate(files):
                if file_path in file.name:
                    print(file.name)
                    down_name = "temp_data/" + file.name.split('/')[-1][:-3] + "ogg"
                    cvt_name = "data/true/" + file.name.split('/')[-1][:-3] + "wav"
                    user_storage.child(file.name).download(down_name)
                    os.system("ffmpeg -i "+ down_name + " -ac 1 -ab 256 -ar 16000 " + cvt_name)
            os.system("rm -r temp_data")

            level = np.arange(0, 17, 2)
            train_files = []
            for name in os.listdir('data/true/'):
                full_path = 'data/true/' + name
                train_files.append(full_path)

            for file in train_files:
                y, sr = librosa.load(file, sr=16000)
                for l in level:
                    y_third = librosa.effects.pitch_shift(y, sr, n_steps=l)
                    path = file[:-4] + '_' + str(l) + '_' +'_speed.wav'
                    y_third = (y_third / np.max(np.abs(y_third))*32767).astype('int16')
                    print(path)
                    write(path, 16000, y_third)

            train_files = []
            for name in os.listdir('data/true/'):
                full_path = 'data/true/' + name
                train_files.append(full_path)

            for file in train_files:
                rate, data = wav.read(file)
                for i in range(4):
                    start_ = int(np.random.uniform(-1200,2400))
                    if start_ >= 0:
                        wav_time_shift = np.int16(np.r_[data[start_:], np.random.uniform(-0.001,0.001, start_)])
                    else:
                        wav_time_shift = np.int16(np.r_[np.random.uniform(-0.001,0.001, -start_), data[:start_]])
                    wav_time_shift = (wav_time_shift / np.max(np.abs(wav_time_shift))*32767).astype('int16')
                    path = file[:-4] + '_' + str(i) + '_' +'_shift.wav'
                    print(path)
                    write(path, 16000, wav_time_shift)


            dataset_path = 'data'
            all_targets = [name for name in os.listdir(dataset_path) if isdir(join(dataset_path, name))]
            target_list = all_targets
            perc_keep_samples = 1.0 # 1.0 is keep all samples
            sample_rate = 8000
            num_mfcc = 16
            len_mfcc = 16

            filenames = []
            y = []
            for index, target in enumerate(target_list):
                print(join(dataset_path, target))
                filenames.append(os.listdir(join(dataset_path, target)))
                y.append(np.ones(len(filenames[index])) * index)
            filenames = [item for sublist in filenames for item in sublist]
            y = [item for sublist in y for item in sublist]

            filenames_y = list(zip(filenames, y))
            random.shuffle(filenames_y)
            filenames, y = zip(*filenames_y)
            print("Number of Files: ", len(filenames))

            x_train, y_train, prob = extract_features(filenames, y)

            x_train = np.array(x_train)
            x_train = x_train.reshape(
                x_train.shape[0], 
                x_train.shape[1], 
                x_train.shape[2], 
                1)
            y_train = np.array(y_train)
            print(y_train[:5])
            false_feature_sets = np.load('false_data.npz')
            #print(false_feature_sets.files)
            x_train_false = false_feature_sets['x_train']
            y_train_false = false_feature_sets['y_train']

            x_train_false = x_train_false.reshape(
                                    x_train_false.shape[0], 
                                    x_train_false.shape[1], 
                                    x_train_false.shape[2], 
                                    1)
            y_train_false = np.array(y_train_false)
            print(x_train.shape, x_train_false.shape)
            print(y_train.shape, y_train_false.shape)
            x_train = np.vstack([x_train, x_train_false])
            y_train = np.r_[y_train, y_train_false]
            print(x_train.shape, y_train.shape)
            s = np.arange(x_train.shape[0])
            np.random.shuffle(s)
            x_train = x_train[s]
            y_train = y_train[s]

            model = models.load_model('wake_word_stop_model.h5')

            model.pop()
            model.pop()
            model.pop()

            for l in model.layers:
                l.trainable = False
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(1, activation='sigmoid'))

            model.summary()



            model.compile(loss='binary_crossentropy', 
                          optimizer=tf.keras.optimizers.Adam(), 
                          metrics=['acc', tf.keras.metrics.BinaryAccuracy(threshold=0.75)])
            history = model.fit(x_train, 
                                y_train, 
                                epochs=10, 
                                batch_size=32)
            converter = lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            open('new.tflite', 'wb').write(tflite_model)
            interpreter = Interpreter('new.tflite')
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            new_ent.clear()


        elif device_mode.queue[-1]=='security':
            rainbow.put(1)
            fft = np.fft.fft(frame)
            magnitude = np.abs(fft)
            left_spectrum = magnitude[:int(len(magnitude)/2)]
            max_freq = np.argmax(left_spectrum) * 16000 / len(fft)
            print(max_freq)

            if max_freq > 600 :
                trig_ent.set()
                rainbow.put(0)
                start_time = time.time()
                print("abnormal sound detect!")
                print("idx : ", idx,", angle_idx :", angle_list[-1][1])
                time.sleep(0.15)
                angles = []
                for angle, a_idx in list(angle_list)[::-1]:
                    if idx >= a_idx:
                        if (idx - a_idx) < 25:
                            angles.append(angle)
                    else:
                        if (idx - a_idx) < -4950:
                            angles.append(angle)
                print("angles:",angles)
                if len(angles) == 0:
                    duration = int(time.time() - start_time)
                    pass
                else:
                    mode_angle = mode(angles).mode
                    if mode_angle:
                        mode_angle = int(mode_angle[0])
                        if mode_angle < 85:
                            mode_angle = 0
                        elif mode_angle > 95 and mode_angle < 180:
                            mode_angle = 180
                        mode_angle = abs(mode_angle - 180)
                        loading.put(1)
                        fast_turn_motor(pin, mode_angle, pre_angle, min_pulse_ms,2)
                        loading.put(0)
                        duration = int(time.time() - start_time)
                        pre_angle = mode_angle
                        print("mode: ", mode_angle)
                    else:
                        duration = int(time.time() - start_time)
                        pass
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
            for i in range(0, duration *2):
                tc_audio.get()
            for i in range(0,10):
                frame, idx = tc_audio.get()
                frame_5.extend(frame)
                if len(frame_5) == 80000  :
                 
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
                    os.remove(audio_filename)
    
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
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = output_data[0][0]
            #prediction = net(sample)
            print("idx : ", idx,", angle_idx :", angle_list[-1][1], ", pred : ", prediction)
            # print("idx : ", idx,", angle_idx :", angle_list[-1][1])
            if prediction>0.50:
#write(str(it) + '.wav', 8000, resample)
# it += 1
                trig_ent.set()
                rainbow.put(0)
                time.sleep(0.15)
                print("trigger!! idx :",idx, ", angle_idx :", angle_list[-1][1])
                angles = []
                # print("angle_list:", angle_list)
                for angle, a_idx in list(angle_list)[::-1]:
                    if idx >= a_idx:
                        if (idx - a_idx) < 50:
                            angles.append(angle)
                    else:
                        if (idx - a_idx) < -4950:
                            angles.append(angle)

                print("angles:",angles)
                if len(angles) == 0:
                    rainbow.put(1)
                    continue
                mode_angle = mode(angles).mode
                if mode_angle:
                    mode_angle = int(mode_angle[0])
                else:
                    rainbow.put(1)
                    continue
                print("pre_mode:", mode_angle)
                if mode_angle < 85:
                    mode_angle = 0
                elif mode_angle > 95 and mode_angle < 180:
                    mode_angle = 180
                mode_angle = abs(mode_angle - 180)
                print("after_mode:", mode_angle)
                loading.put(1)
                turn_motor(pin, mode_angle, pre_angle, min_pulse_ms,2)
                loading.put(0)
                rainbow.put(1)
                pre_angle = mode_angle

        time.sleep(0.1)



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

def fast_turn_motor(pin, angle, pre_angle, min_pulse_ms, step):
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
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')        # fourcc : 동영상 압축 코덱을 표현하는 4-문자코드
        fourcc = cv2.VideoWriter_fourcc(*'jpeg')        
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # webcam_filename = now + ".avi"
        webcam_filename = now + ".mov"
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
        os.remove(webcam_filename)

        webcam_url = storage.child(path_on_cloud).get_url(webcam_filename)
        print("webcam download")

    return webcam_url






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
    smtp.sendmail("lge000325@gmail.com", email_id, msg.as_string())     # 받는 자의 이메일 email_id
    print(msg.as_string())
    print("send email!")
    smtp.quit()


