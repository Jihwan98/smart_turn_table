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

from collections import Counter
from sLocalization.ssl_thread import *

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
pin = 4
min_pulse_ms = 0.52



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


def tc_thread(tc_audio, device_mode,email_id,loading,rainbow):
    at = get_at()
    dict_label = {
        0:at[-1], 1:at[0], 2:at[1], 3:at[2], 4:at[3], 5:at[4], 6:at[5],
        7:at[6], 8:at[7], 9:at[8], 10:at[9], 11:at[10], 12:at[11], 13:at[12]
    }
    net = SounDNet()
    net.load_state_dict(torch.load('sLocalization/weight/499.pth', map_location=torch.device('cpu')))
    net.eval()

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
        frame, ssl_frame, idx = tc_audio.get()
        if device_mode.queue[-1]=='security':
            rainbow.put(1)
            fft = np.fft.fft(frame)
            magnitude = np.abs(fft)
            left_spectrum = magnitude[:int(len(magnitude)/2)]
            max_freq = np.argmax(left_spectrum) * 16000 / len(fft)
            print(max_freq)

            if max_freq > 600 :
                rainbow.put(0)
                start_time = time.time()
                print("abnormal sound detect!")
                print("idx : ", idx)
                
                ssl_frames = np.vsplit(ssl_frame[320:], 12)
                magnitude = 0
                angle_frame = 0
                max_idx = 0
                for i, ssl_f in enumerate(ssl_frames):
                    if np.sum(np.abs(ssl_f)) > magnitude:
                        # angle_frame = ssl_f
                        magnitude = np.sum(np.abs(ssl_f))
                        max_idx = i
                
                angle_frame = ssl_frame
                print('max_idx :', max_idx)
                """# 3
                if max_idx == 0:
                    angle_frame = np.concatenate((ssl_frames[0], ssl_frames[1], ssl_frames[2]), axis=0)
                elif max_idx == 11:
                    angle_frame = np.concatenate((ssl_frames[9], ssl_frames[10], ssl_frames[11]), axis=0)
                else: 
                    angle_frame = np.concatenate((ssl_frames[max_idx-1], ssl_frames[max_idx], ssl_frames[max_idx+1]), axis=0)
                """


                """ 5
                if max_idx <= 1:
                    angle_frame = np.concatenate((ssl_frames[0], ssl_frames[1], ssl_frames[2], ssl_frames[3], ssl_frames[4]), axis=0)
                elif max_idx >= 10:
                    angle_frame = np.concatenate((ssl_frames[7], ssl_frames[8], ssl_frames[9], ssl_frames[10], ssl_frames[11]), axis=0)
                else: 
                    angle_frame = np.concatenate((ssl_frames[max_idx-2], ssl_frames[max_idx-1], ssl_frames[max_idx], ssl_frames[max_idx+1], ssl_frames[max_idx+2]), axis=0)
                """
                mic_sum = np.sum(np.abs(angle_frame), axis=0)
                print("mic_sum ---------\n", mic_sum, "\n--------------------")
                max_mic = np.argmax(mic_sum)
                print("max_mic :", max_mic+1)

                angle = int(np.around(180 - (max_mic - 1) * (360 / 7)))
                
                if angle > 180:
                    angle = 180
                elif angle < 0:
                    angle = 0
                else:
                    pass

                print("angle:",angle)
                loading.put(1)
                fast_turn_motor(pin, angle, pre_angle, min_pulse_ms,2)
                loading.put(0)
                duration = int(time.time() - start_time)
                pre_angle = angle
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
                frame, ssl_frame, idx = tc_audio.get()
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
            print("idx : ", idx, ", pred : ", prediction)
            if prediction>0.50:
                rainbow.put(0)
                """
                ssl_frames = np.vsplit(ssl_frame, 25)
                magnitude = 0
                angle_frame = 0
                max_idx = 0
                for i, ssl_f in enumerate(ssl_frames):
                    if abs(np.sum(ssl_f)) > magnitude:
                        # angle_frame = ssl_f
                        magnitude = abs(np.sum(ssl_f))
                        max_idx = i
                               
                print('max_idx :', max_idx)
                if max_idx == 0:
                    angle_frame = np.concatenate((ssl_frames[0], ssl_frames[1], ssl_frames[2]), axis=0)
                elif max_idx == 24:
                    angle_frame = np.concatenate((ssl_frames[22], ssl_frames[23], ssl_frames[24]), axis=0)
                else:
                    angle_frame = np.concatenate((ssl_frames[max_idx-1], ssl_frames[max_idx], ssl_frames[max_idx+1]), axis=0)
                """
                angle_frame = np.vsplit(ssl_frame, ( 640 * 7, 640 * 18 ))[1]

                sample = torch.tensor(angle_frame).view(-1,640,8).permute(0, 2, 1).contiguous().float()
                out_re, out_an = net(sample)
                
                angle_list = []
                for re, an in zip(out_re, out_an):
                    re_ = torch.tensor(re).view(1, 14)
                    an_ = torch.tensor(an).view(1, 14)
                    pred_angle, pred_idx = get_angle_error(an_, re_, dict_label)
                    if pred_idx > 0 and pred_idx < 7:
                        continue

                    angle = int(pred_angle.detach().numpy())
                    resi_angle = angle % 5
                    if resi_angle >= 3:
                        angle = angle + (5 - resi_angle)
                    else:
                        angle = angle - resi_angle
                    angle_list.append(angle)
                print("angle_list :", angle_list)
                if Counter(angle_list).most_common(1)[0][1] == 1:
                    angle = angle_list[1]
                else:
                    angle = Counter(angle_list).most_common(1)[0][0]

                print("angle :",angle)
                if angle < 85:
                    angle = 0
                elif angle > 95 and angle < 180:
                    angle = 180
                after_angle = abs(angle - 180)
                print("after_angle :", after_angle)
                loading.put(1)
                turn_motor(pin, after_angle, pre_angle, min_pulse_ms,2)
                loading.put(0)
                rainbow.put(1)
                pre_angle = after_angle

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


