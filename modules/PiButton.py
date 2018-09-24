# https://raspberrypi.stackexchange.com/questions/42807/threading-with-gpio-and-buttons

from threading import Thread
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

class Button:
    def __init__(self, channel):
        threading.Thread.__init__(self)
        self._pressed = False
        self.channel = channel
        self.count = 0
        GPIO.setup(self.channel, GPIO.IN)
        self.deamon = True
        self.start()

    def run(self):
        previous = None
        while 1:
            current = GPIO.input(self.channel)
            time.sleep(0.01)

            if current is False and previous is True:
                self._pressed = True
                if self.count > 10:
                    self.count = 0
                else:
                    self.count += 1

                while self._pressed:
                    time.sleep(0.05)

            previous = current

    def clearCount(self):
        self.count = 0
