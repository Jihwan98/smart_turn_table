from matrix_lite import led 
import time
from math import pi,sin
import collections
import queue

def led_loading(switch):
    everloop = ['black'] * led.length
    everloop[0] = {'b':100}
    while True:
        everloop.append(everloop.pop(0))
        led.set(everloop)
        time.sleep(0.050)
        if switch.queue[-1] == 0:
            led.set('black')
            break

def twinkle(color=None):
    if color == 'blue':
        colors = ['blue', 'black']
    elif color == 'red':
        colors = ['red', 'black']
    elif color == 'yellow':
        colors = ['yellow', 'black']
    else:
        colors = ['white', 'black']
    for i in range(0,4):
        led.set(colors[i%2])
        time.sleep(1)


def rainbow_led(rainbow):
    everloop = ['black'] * led.length

    ledAdjust = 0.0
    if len(everloop) == 35:
        ledAdjust = 0.51 # MATRIX Creator
    else:
        ledAdjust = 1.01 # MATRIX Voice

    frequency = 0.375
    counter = 0.0
    tick = len(everloop) - 1

    while True:
        # Create rainbow
        for i in range(len(everloop)):
            r = round(max(0, (sin(frequency*counter+(pi/180*240))*155+100)/10))
            g = round(max(0, (sin(frequency*counter+(pi/180*120))*155+100)/10))
            b = round(max(0, (sin(frequency*counter)*155+100)/10))

            counter += ledAdjust

            everloop[i] = {'r':r, 'g':g, 'b':b}

       # Slowly show rainbow
        if tick != 0:
            for i in reversed(range(tick)):
                everloop[i] = {}
            tick -= 1

        led.set(everloop)

        time.sleep(.035)
        if rainbow.queue[-1] == 0:
            led.set('black')
            break

def led_thread(internet_ent, internet, loading, device_mode, rainbow):
    past_internet = 0
    past_mode = 'normal'
    internet_ent.wait()
    while True:
        internet_status = internet.queue[-1]
        now_mode = device_mode.queue[-1]
        if internet_status != past_internet:
            if internet_status == 1:
                twinkle('blue')
            else:
                twinkle('red')
        past_internet = internet_status

        if loading.queue[-1] == 1:
            rainbow.put(0)
            led_loading(loading)
        loading_status = loading.queue[-1]
        if loading.qsize() > 5000:
            loading.queue.clear()
            loading.put(loading_status)
    
        if now_mode != past_mode:
            twinkle()
        past_mode = now_mode

        if rainbow.queue[-1] == 1:
            rainbow_led(rainbow)
        rainbow_status = rainbow.queue[-1]
        if rainbow.qsize() > 5000:
            rainbow.queue.clear()
            rainbow.put(rainbow_status)


        time.sleep(0.01)
