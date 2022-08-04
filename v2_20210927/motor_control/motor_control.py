import time
from matrix_lite import gpio
pin = 4
min_pulse_ms = 0.5

def delay_time(d):
    return (0.5 * (d-0.5)**2) + 0.025

def turn_motor(pin, angle, pre_angle, min_pulse_ms, step):
    step = step if angle > pre_angle else step * (-1)
    dis = abs(pre_angle - angle)

    if dis > 30:
        for i in range(pre_angle, angle+2, step):
            gpio.setServoAngle({
                    'pin': pin,
                    'angle': i,
                    'min_pulse_ms': min_pulse_ms,
            })
            time.sleep(delay_time(abs(i-pre_angle)/dis))

    else:
        for i in range(pre_angle, angle+2, step):
            gpio.setServoAngle({
                    'pin': pin,
                    'angle': i,
                    'min_pulse_ms' : min_pulse_ms,
            })
            time.sleep(0.1)

if __name__ == "__main__":
    pre_angle = 0
    while True:
        angle = int(input("angle : "))
        turn_motor(pin, angle, pre_angle, min_pulse_ms, 2)
        pre_angle = angle
