import time
uptime = time.time()
print('[LOG] Start Setup...')

import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from picamera import PiCamera
from picamera.array import PiRGBArray
import modules.PiVideoStream as PiVideoStream
import modules.lcddriver as lcddriver
import modules.Stepper as Stepper
import modules.PiButton as PiButton
import RPi.GPIO as GPIO
from random import shuffle

NUM_PLAYER = 1
ROUND_TIME = 30
#EMOTIONS = {3:"happy", 5:"surprise", 6:"neutral", 0:"angry", 1:"disgust", 4:"sad", 2:"fear"} # order by probability, without order take emotion_labels
EASY_EMOTIONS = {6:'neutral', 3:"happy", 5:"surprise", 0:"angry"}
next_emotion_t = 10 # time until next random emotion in game
scores = []
level = 0.51 # emotion prob. needs to be higher to score
EASY_MODE = True

USE_CAM = True # If false, loads video file source
USE_THREAD = True

# start button thread
button = PiButton.Button(37)

# set up stepper
bounds = 35
stepper = Stepper.Stepper(bounds)
step_scale = (bounds * 2) / 100

# hyper-parameters for bounding box shape
emotion_offsets = (20, 40)

# parameters for loading data and images
emotion_model_path = './models/mini_XCEPTION_KDEF.hdf5' #fer2013_mini_XCEPTION.119-0.65.hdf5' # emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)
#print(emotion_classifier.fit) # not working cause of optimizer warning

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Select video or webcam feed
cap = None
if USE_CAM:
    if USE_THREAD:
        vs = PiVideoStream.PiVideoStream().start()
        time.sleep(1)
    else:
        camera = PiCamera()
        camera.resolution = (320, 240)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(320, 240))
        cap = camera.capture_continuous(rawCapture, format="bgr",
        	use_video_port=True)
        time.sleep(1)
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4')

# init lcd display
lcd = lcddriver.lcd()

print('[LOG] Setup finished in: {}'.format(round(time.time() - uptime), 2))

# Intro
lcd.lcd_display_string_animated_mid('EMOTRONOM', 1, 0.2)
time.sleep(1)
lcd.lcd_display_string_animated('Emotion Detector', 2, 0.1)
time.sleep(3)
lcd.lcd_clear()

# Skip Tutorial
button.clearCount()
lcd.lcd_display_string_animated_mid('Press Button', 1, 0.1)
lcd.lcd_display_string_animated_mid('for Tutorial', 2, 0.1)
time.sleep(5)
lcd.lcd_clear()

# Tutorial
if button.count != 0:
    lcd.lcd_display_string_animated_mid('TUTORIAL', 1, 0.1)
    time.sleep(1)
    lcd.lcd_display_string_long('There are seven emotions. Neutral, Happy, Angry, Sad, Disgust, Fear and Surprise.', 2, 0.1)
    time.sleep(0.5)
    lcd.lcd_display_string_long('Guess the emotion by facial expressions. Then to keep the amplitude high.', 2, 0.1)
    time.sleep(0.5)
    lcd.lcd_display_string_long('Who perseveres the longest, is the KING.', 2, 0.1)
    time.sleep(0.5)
    lcd.lcd_clear()
    lcd.lcd_display_string_animated_mid('    Have Fun    ', 2, 0.1)
    time.sleep(1)
    lcd.lcd_clear()
    button.clearCount()

# Select Players
lcd.lcd_clear()
lcd.lcd_display_string_animated_mid('Number of Player', 1, 0.1)
start_t, diff_t = time.time(), 0
while diff_t < 9:
    diff_t = time.time() - start_t
    lcd.lcd_display_string(str(button.count + 1), 2)
NUM_PLAYER = button.count + 1
lcd.lcd_clear()
button.clearCount()

# Guess Game
lcd.lcd_display_string_animated_mid('Guess Game', 1, 0.1)
lcd.lcd_display_string_animated_mid('Get in Position!', 2 , 0.1)

for player in range(NUM_PLAYER):

    stepper.calibrate()
    stepper.LEFT_TURN(bounds)

    if EASY_MODE:
        copy_emotions = EASY_EMOTIONS.copy()
    else:
        copy_emotions = emotion_labels.copy()
    keys = list(copy_emotions.keys())
    shuffle(keys)
    mixed_emotions = [(key, copy_emotions[key]) for key in keys]
    wanted_emotion = mixed_emotions.pop()
    print('[INFO] First wanted emotion: ' + str(wanted_emotion))

    start_t = last_t = time.time()
    score_t = diff_t = 0

    lastProb = None
    loopCount = 0
    predictCount = 0
    pipeline_t = []

    while diff_t < ROUND_TIME or lastProb == None:

        if round(diff_t) > next_emotion_t and predictCount > 0:
            next_emotion_t += 10
            wanted_emotion = mixed_emotions.pop()
            print('[INFO] Next wanted emotion: ' + str(wanted_emotion))

        # get image from picamera
        if USE_CAM and USE_THREAD:
            bgr_image = vs.read()
        else:
            ret, bgr_image = cap.read()
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
    			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # select dominating face
        curr, major_face = 0, None
        for i, (x, y, w, h) in enumerate(faces):
            if w * h > curr:
                curr = w * h
                major_face = i

        # predict emotions
        if major_face != None:
            predict_t = time.time()
            face_coordinates = faces[major_face]
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            wanted_emotion_prob = emotion_prediction[0, wanted_emotion[0]]
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            print('[INFO] predicted emotion: ' + emotion_text + ' / ' + str(round(emotion_probability,2)))
            #print('[INFO] wanted probability: ' + str(wanted_emotion_prob))

            if predictCount > 1:
                if wanted_emotion_prob > level:
                    score_t = score_t + (time.time() - last_t)
                lcd.lcd_display_string(str(round(score_t, 2)), 2)
                step = round(step_scale * (100 * wanted_emotion_prob) - bounds - stepper.getPos(), 1)
                if wanted_emotion_prob > lastProb:
                    stepper.RIGHT_TURN(step)
                else:
                    stepper.LEFT_TURN(abs(step))
                #print('[INFO] StepperPos: ' + str(stepper.getPos()) + ' new step: ' + str(step) + ' lastProb ' + str(lastProb))

            if predictCount == 0:
                print('[INFO] Warmup Pipeline Time ' + str(time.time() - last_t))
                # Countdown
                lcd.lcd_clear()
                lcd.lcd_display_string_animated_mid('Starts in', 1, 0.1)
                start_t, diff_t = time.time(), 0
                while diff_t < 5:
                    diff_t = time.time() - start_t
                    elapsed_t = ' ' * 7 + str(5-round(diff_t))
                    lcd.lcd_display_string(elapsed_t, 2)
                lcd.lcd_clear()
                start_t = last_t = time.time()
                score_t = diff_t = 0 
                lcd.lcd_display_string('Player {}:'.format(player),1)
                lcd.lcd_display_string(str(score_t),2)

            lastProb = wanted_emotion_prob
            predictCount += 1

            #print('[LOG] Full Pipeline Time ' + str(time.time() - last_t))
            pipeline_t.append(time.time() - last_t)

        else:
            # if no face detected
            stepper.LEFT_TURN(1)

        diff_t = time.time() - start_t
        last_t = time.time()

        loopCount += 1

        if button.count != 0:
            button.clearCount()
            print('[INFO] Break by Button!')
            break

    #print('[LOG] Loops per second: {}'.format(loopCount / ROUND_TIME)) # not right anymore
    #print('[LOG] Prediction count: {}'.format(predictCount))
    print('[LOG] Average Full Pipeline Time ' + str(sum(pipeline_t) / predictCount))
    pipeline_t.clear()

    # displaying player score
    lcd.lcd_display_string_animated_mid('STOP!', 1, 0.05)
    time.sleep(1)
    scores.append(score_t)
    next_emotion_t = 10
    lcd.lcd_display_string_animated_mid('PL {} score is'.format(player), 1, 0.1)
    time.sleep(3)
    if player+1 < NUM_PLAYER:
        lcd.lcd_clear()
        lcd.lcd_display_string_animated_mid('Next Player', 1, 0.1)
        lcd.lcd_display_string_animated_mid('get ready...', 2, 0.1)

# displaying results
lcd.lcd_clear()
time.sleep(0.5)
lcd.lcd_display_string_animated_mid('Game done!', 1, 0.1)
stepper.calibrate()
time.sleep(2)
lcd.lcd_display_string_animated_mid('Winner is PL {}'.format(scores.index(max(scores))), 1, 0.1)
lcd.lcd_display_string_animated_mid('with time {}'.format(round(max(scores),2)), 2, 0.1)
time.sleep(10)

# TODO
# again?

# cleanup
lcd.lcd_clear()
lcd.lcd_backlight('off')
button.stop()
if USE_THREAD and USE_CAM:
    vs.stop()
else:
    cap.release()
    rawCapture.close()
    camera.close()
cv2.destroyAllWindows()
