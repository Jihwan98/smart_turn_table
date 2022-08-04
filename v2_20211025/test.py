"""
Created on Fri Aug 20 16:31:08 2021

@author: Ko Jungbeom
"""

from threading import Thread, Lock, Event
import collections
import time
import sys
#sys.path.append('ssl/')
#sys.path.append('tc/')

from matrix_lite import gpio, led

from bluetooth.bluetooth import *
from sLocalization.ssl_thread2 import *
from tc.tc import *
from audio.audio import *
from ssl_audio.ssl_audio import *
from tc_audio.tc_audio import *
from checkInternet.checkInternet import *
from led_thread.led_thread import *
from firebase.firebase_thread import *

from ctypes import *
import queue

pin = 4
min_pulse_ms = 0.5

if __name__ == '__main__' :
    try:    
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            pass
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        asound = cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        
        """
        vad = VADAudio(
                aggressiveness=2,
                input_rate=16000
        )
        """
        
        rainbow = queue.Queue()#
        internet = queue.Queue()#
        loading = queue.Queue()#
        audio2TC_audio = queue.Queue()#
        audio2SSL_audio = queue.Queue()#
        TC_audio2TC = queue.Queue()#
        SSL_audio2SSL = queue.Queue()#
        device_mode = queue.Queue()#
        device_mode.put('normal')
        loading.put(0)
        rainbow.put(0)
        
        angle_list = collections.deque()#
        to_tc_deq = collections.deque()#
        email_deq = collections.deque()#

        trig_ent = Event()
        internet_ent = Event()
        check_internet_ent = Event()
        new_ent = Event()
        start_bluetooth = Thread(target=start_bluetooth)
        bluetooth_thread = Thread(target=bluetooth_thread, args=(email_deq,loading,))
        internet_thread = Thread(target=checkInternet_thread, args=(internet, check_internet_ent, internet_ent,))
        led_thread = Thread(target=led_thread, args=(check_internet_ent, internet, loading, device_mode, rainbow,))
        get_command_thread = Thread(target=get_command_thread, args=(internet_ent,to_tc_deq, email_deq, device_mode,))
        audio_thread = Thread(target=audio_thread, args=(audio2SSL_audio,audio2TC_audio, loading,rainbow, device_mode,))
        tc_audio = Thread(target=tc_audio_thread, args=(audio2TC_audio, TC_audio2TC, device_mode,))
        ssl_audio = Thread(target=ssl_audio_thread, args=(audio2SSL_audio, SSL_audio2SSL,))
        
        tc = Thread(target=tc_thread, args=(TC_audio2TC, angle_list, device_mode,email_deq,loading,rainbow, trig_ent, new_ent,))
        ssl = Thread(target=ssl_thread, args=(SSL_audio2SSL, angle_list, trig_ent,))
        

        gpio.setServoAngle({
                'pin': pin,
                'angle': 90,
                'min_pulse_ms': min_pulse_ms,
        })
        
        start_bluetooth.daemon=True
        bluetooth_thread.daemon=True
        internet_thread.daemon=True
        led_thread.daemon=True
        get_command_thread.daemon=True
        audio_thread.daemon=True
        tc_audio.daemon=True
        ssl_audio.daemon=True
        tc.daemon=True
        ssl.daemon=True
        
        start_bluetooth.start()
        bluetooth_thread.start()
        internet_thread.start()
        led_thread.start()
        get_command_thread.start()
        audio_thread.start()
        tc_audio.start()
        ssl_audio.start()
        print('Functional Thread Join...')
        tc.start()
        ssl.start()
        
        time.sleep(4)
       
        start_bluetooth.join()
        bluetooth_thread.join()
        internet_thread.join()
        led_thread.join()
        get_command_thread.join()
        audio_thread.join()
        tc_audio.join()
        ssl_audio.join()
        tc.join()
        ssl.join()
        
    except Exception as e:
        print("error :",e)

    finally:
        print("Quit SVRT")
        led.set('black')
