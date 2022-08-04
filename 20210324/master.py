import sys
sys.path.append('firebase/')
from threading import Thread, Lock, Event
import collections
import time

from audio.audio_generator import *
from sLocalization.ssl_thread import *
from usb_detect.usb_thread import *
from deepSpeech.ds_thread import *
from firebase.firebase_thread import *
from utils import draw_logo

from ctypes import *



if __name__ == '__main__' :
    
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)

    usb_ent = Event()
    motor_ent = Event()

    tods_deq = collections.deque()
    email_deq = collections.deque()
    angle_deq = collections.deque()
    
    vad = VADAudio(
            aggressiveness=2,
            input_rate=16000
    )

    ds_frames = vad.vad_collector()
    ssl_frames = vad.ssl_read()

    usb_detection = Thread(target=usb_detection_thread, args=(usb_ent, email_deq))
    get_command = Thread(target=get_command_thread, args=(usb_ent, tods_deq, email_deq))
    ds_thread = Thread(target=deepSpeech_thread, args=(ds_frames, tods_deq, angle_deq, motor_ent,))
    ssl_thread = Thread(target=sLocalizer, args=(ssl_frames, angle_deq, motor_ent,))

    print('USE Detector Thread Join...')
    usb_detection.start()
    get_command.start()
    print('DeepSpeech Thread Join...')
    ds_thread.start()
    print('BlankNet Thread Join...')
    ssl_thread.start()
    
    time.sleep(4)
    draw_logo.draw_logo()
    
    usb_detection.join()
    get_command.join()
    ds_thread.join()
    ssl_thread.join()

    
    print("Quit SVRT")

