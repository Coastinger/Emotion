import lcddriver
import time

lcd = lcddriver.lcd()

lcd.lcd_backlight('on')
lcd.lcd_clear()
lcd.lcd_display_string('Hello World', 1)
time.sleep(2)
lcd.lcd_display_string('Hello You', 2)
time.sleep(2)
lcd.lcd_clear()
time.sleep(2)
lcd.lcd_display_string('------MID-------', 1)
time.sleep(1)
lcd.lcd_display_string('FinalTestCounter', 2)
time.sleep(2)
lcd.lcd_clear()
lcd.lcd_display_string('Time elapsed:', 1)
now = time.time()
later = 0
while later < 10:
     later =  time.time() - now
     elapsed = str(round(later,2))
     lcd.lcd_display_string(elapsed,2)
time.sleep(2)

lcd.lcd_backlight('off')
