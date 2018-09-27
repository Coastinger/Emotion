import threading
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

class Button(threading.Thread):
    def __init__(self, channel):
        threading.Thread.__init__(self)
        self.channel = channel
        self.count = 0
        self.running = True
        GPIO.setup(self.channel, GPIO.IN, pull_up_down=GPIO.PUD_UP )
        self.deamon = True
        self.start()

    def run(self):
        while self.running:
            if GPIO.input(self.channel) == GPIO.LOW:
                if self.count < 10:
                    self.count += 1
                else:
                    self.count = 0
            time.sleep(0.3)

    def clearCount(self):
        self.count = 0

    def stop(self):
        self.running = False
