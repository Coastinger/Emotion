import time
import RPi.GPIO as GPIO

#GPIO.setmode(GPIO.BOARD)
#GPIO.setwarnings(False)

# https://www.python-forum.de/viewtopic.php?t=36037

A, B, C, D = 7, 11, 13, 15
full_circle = 510.0

class Stepper:

    def __init__(self):
        GPIO.setup(A, GPIO.OUT)
        GPIO.setup(B, GPIO.OUT)
        GPIO.setup(C, GPIO.OUT)
        GPIO.setup(D, GPIO.OUT)
        GPIO.output(pin, False)

    def GPIO_SETUP(a,b,c,d):
        GPIO.output(A, a)
        GPIO.output(B, b)
        GPIO.output(C, c)
        GPIO.output(D, d)
        time.sleep(0.001)

    def RIGHT_TURN(deg):
        degree = full_circle/360*deg
        GPIO_SETUP(0,0,0,0)

        while degree > 0.0:
            GPIO_SETUP(1,0,0,0)
            GPIO_SETUP(1,1,0,0)
            GPIO_SETUP(0,1,0,0)
            GPIO_SETUP(0,1,1,0)
            GPIO_SETUP(0,0,1,0)
            GPIO_SETUP(0,0,1,1)
            GPIO_SETUP(0,0,0,1)
            GPIO_SETUP(1,0,0,1)
            degree -= 1

    def LEFT_TURN(deg):
        degree = full_circle/360*deg
        GPIO_SETUP(0,0,0,0)

        while degree > 0.0:
            GPIO_SETUP(1,0,0,1)
            GPIO_SETUP(0,0,0,1)
            GPIO_SETUP(0,0,1,1)
            GPIO_SETUP(0,0,1,0)
            GPIO_SETUP(0,1,1,0)
            GPIO_SETUP(0,1,0,0)
            GPIO_SETUP(1,1,0,0)
            GPIO_SETUP(1,0,0,0)
            degree -= 1
