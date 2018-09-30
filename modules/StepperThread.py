import threading
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

class Stepper(threading.Thread):

    def __init__(self, bounds, A, B, C, D):
        threading.Thread.__init__(self)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        GPIO.setup(A, GPIO.OUT)
        GPIO.setup(B, GPIO.OUT)
        GPIO.setup(C, GPIO.OUT)
        GPIO.setup(D, GPIO.OUT)
        self.pos = 0
        self.bounds = bounds
        self.full_circle = 510.0
        self.running = True
        self.goTo = 0
        self.start()

    def run(self):
        while self.running:
            diff = self.pos - self.goTo
            if diff >= 1 or diff <= -1:
                # do 1 step
                self.GPIO_SETUP(0,0,0,0)
                if self.goTo < self.pos:
                    # LEFT
                    self.GPIO_SETUP(1,0,0,1)
                    self.GPIO_SETUP(0,0,0,1)
                    self.GPIO_SETUP(0,0,1,1)
                    self.GPIO_SETUP(0,0,1,0)
                    self.GPIO_SETUP(0,1,1,0)
                    self.GPIO_SETUP(0,1,0,0)
                    self.GPIO_SETUP(1,1,0,0)
                    self.GPIO_SETUP(1,0,0,0)
                    self.setPos(-1)
                else:
                    # RIGHT
                    self.GPIO_SETUP(1,0,0,0)
                    self.GPIO_SETUP(1,1,0,0)
                    self.GPIO_SETUP(0,1,0,0)
                    self.GPIO_SETUP(0,1,1,0)
                    self.GPIO_SETUP(0,0,1,0)
                    self.GPIO_SETUP(0,0,1,1)
                    self.GPIO_SETUP(0,0,0,1)
                    self.GPIO_SETUP(1,0,0,1)
                    self.setPos(1)
            time.sleep(0.001)

    def GPIO_SETUP(self, a, b, c, d):
        GPIO.output(self.A, a)
        GPIO.output(self.B, b)
        GPIO.output(self.C, c)
        GPIO.output(self.D, d)
        time.sleep(0.001)

    def setMove(self, pos):
        if pos >= -self.bounds and pos <= self.bounds:
            self.goTo = pos
        else:
            print('[STEPPA] Step out of bounds!')

    def stop(self):
        self.running = False

    def setPos(self, deg):
        self.pos += deg

    def getPos(self):
        return self.pos

    def calibrate(self):
        self.setMove(0)
