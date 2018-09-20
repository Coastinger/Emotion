import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# https://www.python-forum.de/viewtopic.php?t=36037

A, B, C, D = 7, 11, 13, 15
full_circle = 510.0

class Stepper:

    def __init__(self, bounds):
        GPIO.setup(A, GPIO.OUT)
        GPIO.setup(B, GPIO.OUT)
        GPIO.setup(C, GPIO.OUT)
        GPIO.setup(D, GPIO.OUT)
        self.pos = 0
        self.bounds = bounds

    def GPIO_SETUP(self, a, b, c, d):
        GPIO.output(A, a)
        GPIO.output(B, b)
        GPIO.output(C, c)
        GPIO.output(D, d)
        time.sleep(0.001)

    def RIGHT_TURN(self, deg):
        self.setPos(deg)
        degree = full_circle/360*deg
        self.GPIO_SETUP(0,0,0,0)

        while degree > 0.0:
            self.GPIO_SETUP(1,0,0,0)
            self.GPIO_SETUP(1,1,0,0)
            self.GPIO_SETUP(0,1,0,0)
            self.GPIO_SETUP(0,1,1,0)
            self.GPIO_SETUP(0,0,1,0)
            self.GPIO_SETUP(0,0,1,1)
            self.GPIO_SETUP(0,0,0,1)
            self.GPIO_SETUP(1,0,0,1)
            degree -= 1

    def LEFT_TURN(self, deg):
        self.setPos(-deg)
        degree = full_circle/360*deg
        self.GPIO_SETUP(0,0,0,0)

        while degree > 0.0:
            self.GPIO_SETUP(1,0,0,1)
            self.GPIO_SETUP(0,0,0,1)
            self.GPIO_SETUP(0,0,1,1)
            self.GPIO_SETUP(0,0,1,0)
            self.GPIO_SETUP(0,1,1,0)
            self.GPIO_SETUP(0,1,0,0)
            self.GPIO_SETUP(1,1,0,0)
            self.GPIO_SETUP(1,0,0,0)
            degree -= 1

    def setPos(self, deg):
        self.pos = deg

    def getPos(self):
        return self.pos

    def calibrate(self):
        # Return to position 0.
        if self.pos > 0:
           self.LEFT_TURN(self.pos)
        else:
           self.RIGHT_TURN(abs(self.pos))

    def is_valid(self, deg):
        if self.getPos() + deg > self.bounds:
           return False
        return True
