import time
import Stepper

"""
      1  2  3  4  5  6  7  8
      
Pin1  x  x                 x
Pin2     x  x  x
Pin3           x  x  x
Pin4                 x  x  x

"""

bounds = 10

print('Init with bounds ' + str(bounds))
stepper = Stepper.Stepper(bounds)
print('Test LEFT_TURN...')
stepper.LEFT_TURN(9)
time.sleep(0.5)
print('Test RIGHT_TURN...')
stepper.RIGHT_TURN(18)
time.sleep(0.5)
print('Calibrate...')
stepper.calibrate()
print('Test done.')
