from matrix_lite import gpio
import chars2vec
import time
import tensorflow as tf

from trigger.trigger_thread import *

pin = 4
min_pulse_ms = 0.5

def trigger_detect_thread(audio_que, totd_deq, motor_ent, text_angle_deq, text_deq):
    """Classify whether the text is trigger or not and update trigger model

    Args:
        audio_que(que) : que containing audio frames
        totd_deq(deque) : deque to receive a new trigger word from get_command thread
        motor_ent(event) : event object indicate if the motor finish rotation
        text_angle_deq(deq) : deque containing dictionaries({text:angle}) to receive a text and an angle from deepSpeech_thread
        text_deq(deq) : deque to receive text from deepSpeech_thread

    """
    print("Trigger Detector Thread Join...")
    # Matrix Voice GPIO Setup
    gpio.setFunction(4, 'PWM')
    gpio.setMode(4, 'output')

    c2v_model = chars2vec.load_model('eng_50')
    trigger_model = get_updated_model('friend', c2v_model, 'trigger/data/others.txt')
    angle = 0
    pre_angle = 0
    pre_text = ""
    while True:
        if totd_deq:
            print("New Trigger Detected")
            new_trigger = totd_deq.popleft()
            new_trigger = new_trigger.replace(" ", "").lower()
            trigger_model = get_updated_model(new_trigger, c2v_model, 'trigger/data/others.txt')
            print("Finish Train")
        if len(text_angle_deq)== 0:
            time.sleep(0.001)
            continue
        if text_angle_deq[0][text_deq[0]] is None:
            pre_text = text_deq[0]
            text_angle_deq.popleft()
            text_deq.popleft()
        else: 
            angle = text_angle_deq[0][text_deq[0]]
            text = text_deq[0]
            
            if text_deq[0].startswith(pre_text) and len(pre_text) > 1:
                text = text_deq[0][len(pre_text):].strip()

            preds = []
            print("td text:",text)
            for t in text.split(' '):
                if len(t) < 3:
                    continue
                if "" == t:
                    continue
                if "'" in t:
                    continue
                in_t = list([t])
                in_t = c2v_model.vectorize_words(in_t)
                pred = trigger_model.predict(in_t)
                preds.append(pred)
                #preds = [3,4]
            if 1 in preds:
                print("Trigger!!", text)
                motor_ent.clear()
                turn_motor(pin, angle, pre_angle, min_pulse_ms, 2)
                motor_ent.set()
                pre_angle = angle
            #time.sleep(0.001)
            pre_text = text_deq[0]
            text_angle_deq.popleft()
            text_deq.popleft()


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
