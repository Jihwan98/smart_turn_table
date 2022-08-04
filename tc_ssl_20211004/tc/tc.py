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

import cv2

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
pin = 4
min_pulse_ms = 0.55



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
def f_low(y, sr):
    b, a = signal.butter(10, 2000/(sr/2), btype='lowpass')
    yf = signal.lfilter(b, a, y)
    return yf

def gau_filter(frame):
    kernel1d = cv2.getGaussianKernel(5,3)
    ret = cv2.filter2D(frame, -1, kernel1d)
    return ret

def tc_thread(tc_audio, device_mode,email_id,loading,rainbow,trigword_deq):
    at = get_at()
    dict_label = {
        0:at[-1], 1:at[0], 2:at[1], 3:at[2], 4:at[3], 5:at[4], 6:at[5],
        7:at[6], 8:at[7], 9:at[8], 10:at[9], 11:at[10], 12:at[11], 13:at[12]
    }
    ssl_net = SounDNet()
    # ssl_net.load_state_dict(torch.load('sLocalization/weight/280.pth', map_location=torch.device('cpu')))
    ssl_net.load_state_dict(torch.load('sLocalization/weight/166.pth', map_location=torch.device('cpu')))
    # ssl_net.load_state_dict(torch.load('sLocalization/weight/284.pth', map_location=torch.device('cpu')))
    ssl_net.eval()

    gpio.setFunction(4, 'PWM')
    gpio.setMode(4, 'output')
    sample_rate = 8000
    num_mfcc = 16
    len_mfcc = 16
    resample_size = 8000
    angles = []
    pre_angle = 90
    it = 0
    abc = 0
    net = models.load_model('tc/v1_eddy.h5')
    model_type = "h5"
    """
    interpreter = Interpreter('tc/v0.1_Eddy_TC_20211002.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_type = "tflite"
    """
    while True:
        frame, ssl_frame, idx = tc_audio.get()
        if len(trigword_deq) != 0:
            rainbow.put(0)
            loading.put(1)
            new_trigword = trigword_deq.popleft()
            if new_trigword == "siriya":
                net = models.load_model('tc/siri.h5')
                model_type = "h5"
                print("trigger model : siri")
                device_mode.put(device_mode.queue[-2])
            elif new_trigword == "hobanah":
                net = models.load_model('tc/v0_hoban.h5')
                model_type = "h5"
                print("trigger model : hoban")
                device_mode.put(device_mode.queue[-2])
            else:
                interpreter = Interpreter('tc/v0.1_Eddy_TC_20211002.tflite')
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                model_type = "tflite"
                print("trigger model : eddy")
                device_mode.put(device_mode.queue[-2])
            loading.put(0)
            rainbow.put(1)


        user_id = email_id[-1]
        if device_mode.queue[-1]=='security':
            rainbow.put(1)
            fft = np.fft.fft(frame)
            magnitude = np.abs(fft)
            left_spectrum = magnitude[:int(len(magnitude)/2)]
            max_freq = np.argmax(left_spectrum) * 16000 / len(fft)
            print(max_freq)

            if max_freq > 600:
                rainbow.put(0)
                start_time = time.time()
                print("abnormal sound detect!")
                print("idx : ", idx)
                 
                ssl_max_mean = np.max(np.mean(np.abs(ssl_frame), axis=0))
                if ssl_max_mean < 0.01:
                    ssl_frame = ssl_frame * 0.01 / ssl_max_mean
                    print("ssl scale up")

                a_frame = np.vsplit(ssl_frame[320:], 12)
                """
                magnitude = 0
                angle_frame = 0
                max_idx = 0
                for i, ssl_f in enumerate(ssl_frames):
                    if abs(np.sum(ssl_f)) > magnitude:
                        # angle_frame = ssl_f
                        magnitude = abs(np.sum(ssl_f))
                        max_idx = i
                angle_frame = ssl_frame[320:]

                print('max_idx :', max_idx)
                """

                """
                if max_idx == 0:
                    angle_frame = np.concatenate((ssl_frames[0], ssl_frames[1], ssl_frames[2]), axis=0)
                elif max_idx == 11:
                    angle_frame = np.concatenate((ssl_frames[9], ssl_frames[10], ssl_frames[11]), axis=0)
                else: 
                    angle_frame = np.concatenate((ssl_frames[max_idx-1], ssl_frames[max_idx], ssl_frames[max_idx+1]), axis=0)
                """
                angle_frame = []
                for i in a_frame:
                    if np.mean(np.abs(i)) < 0.0035:
                        continue
                    else:
                        angle_frame.append(i)
                angle_frame = np.array(angle_frame)

                sample = torch.tensor(angle_frame).view(-1,640,8).permute(0, 2, 1).contiguous().float()
                out_re, out_an = ssl_net(sample)
                
                angle_list = []
                for re, an in zip(out_re, out_an):
                    re_ = torch.tensor(re).view(1, 14)
                    an_ = torch.tensor(an).view(1, 14)
                    pred_angle, pred_idx = get_angle_error(an_, re_, dict_label)
                    if pred_idx > 0 and pred_idx < 7:
                        continue
                    angle = int(pred_angle)
                    resi_angle = angle % 10
                    if resi_angle >= 6:
                        angle = angle + (10 - resi_angle)
                    else:
                        angle = angle - resi_angle
                    angle_list.append(angle)
                print("angle_list :", angle_list)
                if len(angle_list) == 0:
                        continue
                angle = Counter(angle_list).most_common(1)[0][0]

                print("angle:",angle)
                if angle < 85:
                    angle = 0
                elif angle > 95 and angle < 180:
                    angle = 180
                after_angle = abs(angle - 180)
                if after_angle < 0:
                    after_angle = 0
                elif after_angle > 180:
                    after_angle = 180
                else:
                    after_angle = after_angle
                loading.put(1)
                fast_turn_motor(pin, after_angle, pre_angle, min_pulse_ms,2)
                loading.put(0)
                duration = int(time.time() - start_time)
                pre_angle = after_angle
                print("after: ", after_angle)
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
            tc_max_mean = np.mean(np.abs(frame))
            if tc_max_mean < 0.008:
                frame = frame * 0.008 / tc_max_mean
                print("tc scale up")
            resample = np.array(signal.resample(frame, resample_size), dtype=np.float32)
            sample = np.float32(calc_mfcc(resample))
            sample = sample.reshape(1, 16, 16, 1)
            if model_type == "tflite":
                interpreter.set_tensor(input_details[0]['index'], sample)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                prediction = output_data[0][0]
            else:
                prediction = net(sample)
            print("idx : ", idx, ", pred : ", prediction)
            if prediction>0.5:
                rainbow.put(0)
                # ssl_frame[:, 1:4] = ssl_frame[:, 1:4] * 1.1
                #b_frame = np.vsplit(ssl_frame, 25)

                # write(str(abc) + ".wav", 16000, frame)
                # abc += 1
                ssl_max_mean = np.max(np.mean(np.abs(ssl_frame), axis=0))
                if ssl_max_mean < 0.01:
                    ssl_frame = ssl_frame * 0.01 / ssl_max_mean
                    print("ssl scale up")
                a_frame = np.vsplit(ssl_frame, ( 640 * 2, 640 * 23))[1]
                """
                gau_frame = []
                for i in range(8):
                    yf = gau_filter(a_frame[:,i])
                    gau_frame.append(yf)
                gau_frame = np.squeeze(np.array(gau_frame))
                gau_frame = np.transpose(gau_frame)
                b_frame = np.vsplit(gau_frame, 11)
                
                """
                """
                low_frame = []
                for i in range(8):
                    yf = f_low(a_frame[:,i], 16000)
                    low_frame.append(yf)
                low_frame = np.transpose(np.array(low_frame))
                """


                b_frame = np.vsplit(a_frame, 21)
                angle_frame = []
                for i in b_frame:
                    if np.mean(np.abs(i)) < 0.0035:
                        continue
                    else:
                        angle_frame.append(i)
                angle_frame = np.array(angle_frame)
                
                
                
                sample = torch.tensor(angle_frame).view(-1,640,8).permute(0, 2, 1).contiguous().float()
                out_re, out_an = ssl_net(sample)
                
                angle_list = []
                for re, an in zip(out_re, out_an):
                    re_ = torch.tensor(re).view(1, 14)
                    an_ = torch.tensor(an).view(1, 14)
                    pred_angle, pred_idx = get_angle_error(an_, re_, dict_label)
                    if pred_idx > 0 and pred_idx < 7:
                        continue

                    angle = int(pred_angle)
                    resi_angle = angle % 10
                    if resi_angle >= 6:
                        angle = angle + (10 - resi_angle)
                    else:
                        angle = angle - resi_angle
                    angle_list.append(angle)
                print("angle_list :", angle_list)
                if len(angle_list) == 0:
                    rainbow.put(1)
                    continue
                angle_dict = {}
                for i in np.unique(np.array(angle_list)):
                    count = 0
                    for j in angle_list:
                        if j >= i - 25 and j <= i + 25:
                            count += 1
                        angle_dict[i] = count
                if np.sum(np.array(list(angle_dict.keys())) == 210) == 1:
                    angle_dict[210] = int(np.ceil(angle_dict[210] / 2))
                sorted_angle = sorted(angle_dict.items(), key=lambda item: item[1], reverse=True)
                print("sorted_angle :", sorted_angle)
                if len(sorted_angle) == 1:
                    angle = sorted_angle[0][0]
                elif len(sorted_angle) == 2:
                    if sorted_angle[0][1] != sorted_angle[1][1]:
                        angle = sorted_angle[0][0]
                    else:
                        angle = round((sorted_angle[0][0] + sorted_angle[1][0]) / 2)
                else:
                    if sorted_angle[0][1] != sorted_angle[1][1]:
                        angle = sorted_angle[0][0]
                    elif sorted_angle[1][1] != sorted_angle[2][1]:
                        angle = round((sorted_angle[0][0] + sorted_angle[1][0]) / 2)
                    else:
                        angle = round((sorted_angle[0][0] + sorted_angle[1][0] + sorted_angle[2][0]) / 3)


                # angle = Counter(angle_list).most_common(1)[0][0]

                print("angle :",angle)
                print("most angle :", Counter(angle_list).most_common(1)[0][0])
                if angle < 85:
                    angle = 0
                elif angle > 95 and angle < 180:
                    angle = 180
                after_angle = abs(angle - 180)
                if after_angle < 0:
                    after_angle = 0
                elif after_angle > 180:
                    after_angle = 180
                else:
                    after_angle = after_angle
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


