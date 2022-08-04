import sys
sys.path.append('firebase/')
from threading import Thread, Lock, Event
import collections
import time
import queue

from sLocalization.ssl_thread import *
from usb_detect.usb_thread import *
from deepSpeech.ds_thread import *
from firebase.firebase_thread import *
from audio_struct.audio_frame import *
from triggerDetect.triggerDetect_thread import *
from utils import draw_logo

from ctypes import *

lock = Lock()

if __name__ == '__main__' :
    
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)

    usb_ent = Event()
    motor_ent = Event()
    audio_ent = Event()

    totd_deq = collections.deque()
    email_deq = collections.deque()
    audio_que = queue.Queue()
    text_angle_deq = collections.deque()
    text_deq = collections.deque()
    
    audio_buffer = Thread(target=audio_thread, args=(audio_que, audio_ent,))
    usb_detection = Thread(target=usb_detection_thread, args=(usb_ent, email_deq))
    get_command = Thread(target=get_command_thread, args=(usb_ent, totd_deq, email_deq))
    ds_thread = Thread(target=deepSpeech_thread, args=(audio_que, audio_ent, text_angle_deq, text_deq,))
    ssl_thread = Thread(target=sLocalizer, args=(audio_que, audio_ent, motor_ent,))
    td_thread = Thread(target=trigger_detect_thread, args=(audio_que, totd_deq, motor_ent, text_angle_deq, text_deq,))

    print("Audio Thread Join...")
    audio_buffer.start()
    print('USB Detector Thread Join...')
    usb_detection.start()
    get_command.start()
    #print('DeepSpeech Thread Join...')
    ds_thread.start()
    #print('BlankNet Thread Join...')
    ssl_thread.start()
    #print('Trigger Detector Thread Join...')
    td_thread.start()
    
    time.sleep(4)
    draw_logo.draw_logo()
    
    audio_buffer.join()
    usb_detection.join()
    get_command.join()
    ds_thread.join()
    ssl_thread.join()
    td_thread.join()

    
    print("Quit SVRT")

