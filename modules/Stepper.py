# Code Base by Technik Tests & Reviews YouTube Channel
# https://www.youtube.com/watch?v=4fHL6BpJrC4
# https://onedrive.live.com/?cid=971235166D23BBC6&id=971235166D23BBC6%21568&parId=971235166D23BBC6%21117&o=OneUp

import threading
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# https://www.python-forum.de/viewtopic.php?t=36037

# A, B, C, D = 7, 11, 13, 15

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

    def RIGHT_TURN(self, deg):
        if self.is_valid(deg):
            #self.running = True
            degree = self.full_circle/360*deg
            self.GPIO_SETUP(0,0,0,0)

            while degree > 0.0 : #and self.running:
                self.GPIO_SETUP(1,0,0,0)
                self.GPIO_SETUP(1,1,0,0)
                self.GPIO_SETUP(0,1,0,0)
                self.GPIO_SETUP(0,1,1,0)
                self.GPIO_SETUP(0,0,1,0)
                self.GPIO_SETUP(0,0,1,1)
                self.GPIO_SETUP(0,0,0,1)
                self.GPIO_SETUP(1,0,0,1)
                degree -= 1
            self.setPos(degree)


    def LEFT_TURN(self, deg):
        if self.is_valid(-deg):
            #self.running = True
            degree = self.full_circle/360*deg
            self.GPIO_SETUP(0,0,0,0)

            while degree > 0.0 : #and self.running:
                self.GPIO_SETUP(1,0,0,1)
                self.GPIO_SETUP(0,0,0,1)
                self.GPIO_SETUP(0,0,1,1)
                self.GPIO_SETUP(0,0,1,0)
                self.GPIO_SETUP(0,1,1,0)
                self.GPIO_SETUP(0,1,0,0)
                self.GPIO_SETUP(1,1,0,0)
                self.GPIO_SETUP(1,0,0,0)
                degree -= 1
            self.setPos(-degree)

    def setMove(self, pos):
        new_pos = pos #self.pos + pos
        if new_pos >= -self.bounds and new_pos <= self.bounds:
            self.goTo = pos
        else:
            print('[STEPPA] Step out of bounds!')

    def stop(self):
        self.running = False

    def stopMove(self):
        self.running = False

    def setPos(self, deg):
        self.pos += deg

    def getPos(self):
        return self.pos

    def calibrate(self):
        # Return to position 0.
        if self.pos > 0:
           self.LEFT_TURN(self.pos)
        else:
           self.RIGHT_TURN(abs(self.pos))

    def is_valid(self, deg):
        new_pos = self.pos + deg
        if new_pos < -self.bounds or new_pos > self.bounds:
            # print('ERROR[STEPER]: Step would be out of bounds!')
            return False
        return True
