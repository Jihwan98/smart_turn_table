from matrix_lite import gpio
import time

# Tell pin 3 to set servo to 90 degrees
gpio.setFunction(4, 'PWM')
gpio.setMode(4, 'output')

for i in range(0,181,2):
    gpio.setServoAngle({
        "pin": 4,
        "angle": i,
        # min_pulse_ms (minimum pulse width for a PWM wave in milliseconds)
        "min_pulse_ms": 0.8,
    })
    time.sleep(0.01)

time.sleep(1)

for i in range(180,-1,-2):
    gpio.setServoAngle({
        "pin": 4,
        "angle": i,
        "min_pulse_ms":0.8,
    })
    time.sleep(0.01)

