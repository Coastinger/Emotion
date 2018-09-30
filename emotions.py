import time
uptime = time.time()
print('[LOG] Start Setup...')

import cv2
import numpy as np
from random import shuffle
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
import modules.PiVideoStream as PiVideoStream
import RPi.GPIO as GPIO
import modules.lcddriver as lcddriver
import modules.StepperThread as StepperThread
import modules.PiStreamAndButton as PiStreamAndButton

USE_VIDEO = False

ENDLESS = True
NUM_PLAYER = 1
ROUND_TIME = 30
#EMOTIONS = {3:"happy", 5:"surprise", 6:"neutral", 0:"angry", 1:"disgust", 4:"sad", 2:"fear"} # order by probability, without order take emotion_labels
EASY_EMOTIONS = {6:'neutral', 3:"happy", 5:"surprise", 0:"angry"}
next_emotion_t = 10 # time until next random emotion in game
scores = []
level = 0.51 # emotion prob. needs to be higher to score
EASY_MODE = True

# set up stepper
bounds = 35
stepper = StepperThread.Stepper(bounds, 7, 11, 13, 15)
step_scale = (bounds * 2) / 100

# hyper-parameters for bounding box shape
emotion_offsets = (20, 40)

# parameters for loading data and images
emotion_model_path = './models/fer2013_mini_XCEPTION.119-0.65.hdf5'
emotion_labels = get_labels('fer2013')

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('./models/lbpcascade_frontalface_improved.xml')
emotion_classifier = load_model(emotion_model_path)
if False:
    # print NN architecture
    for layer in emotion_classifier.layers:
        print(layer.config)
        print(layer.input_shape)
        print(layer.output_shape)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Initiate webcam feed and Button
vs = PiStreamAndButton.PiStreamAndButton().start()
time.sleep(1)

if USE_VIDEO:
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
vs.button.clearCount()
lcd.lcd_display_string_animated_mid('Push Button', 1, 0.1)
lcd.lcd_display_string_animated_mid('for Tutorial', 2, 0.1)
time.sleep(5)
lcd.lcd_clear()
# Tutorial
if vs.button.count != 0:
    lcd.lcd_display_string_animated_mid('TUTORIAL', 1, 0.1)
    time.sleep(1)
    lcd.lcd_clear()
    lcd.lcd_display_string_animated_mid('Seven Emotions',1,0.1)
    time.sleep(0.5)
    lcd.lcd_display_string_animated_mid('Neutral , Happy',2,0.1)
    time.sleep(0.2)
    lcd.lcd_display_string(' ' * 16, 2)
    lcd.lcd_display_string_animated_mid('Angry , Sad',2,0.1)
    time.sleep(0.2)
    lcd.lcd_display_string(' ' * 16, 2)
    lcd.lcd_display_string_animated_mid('Disgust , Fear',2,0.1)
    time.sleep(0.2)
    lcd.lcd_display_string(' ' * 16, 2)
    lcd.lcd_display_string_animated_mid('and Surprise',2,0.1)
    time.sleep(0.5)
    lcd.lcd_clear()
    lcd.lcd_display_string_animated_mid('Make a guess',1,0.1)
    time.sleep(0.1)
    lcd.lcd_display_string_animated_mid('by facial- ',2,0.1)
    time.sleep(0.2)
    lcd.lcd_display_string_animated_mid('expressions',2,0.1)
    time.sleep(1)
    lcd.lcd_clear()
    lcd.lcd_display_string_animated_mid('Goal is to', 1, 0.1)
    lcd.lcd_display_string_animated_mid('last the longest',2,0.1)
    time.sleep(1)
    lcd.lcd_clear()
    lcd.lcd_display_string_animated_mid('Have Fun', 2, 0.1)
    time.sleep(1)
    lcd.lcd_clear()
    vs.button.clearCount()

# Select Players
lcd.lcd_clear()
lcd.lcd_display_string_animated_mid('Player Number', 1, 0.1)
start_t, diff_t = time.time(), 0
while diff_t < 9:
    diff_t = time.time() - start_t
    lcd.lcd_display_string(' ' * 7 + str(vs.button.count + 1), 2)
NUM_PLAYER = vs.button.count + 1
lcd.lcd_clear()
vs.button.clearCount()

while ENDLESS:

    # Select Difficulty
    lcd.lcd_clear()
    lcd.lcd_display_string_animated_mid('Level', 1, 0.1)
    start_t, diff_t = time.time(), 0
    while diff_t < 5:
        diff_t = time.time() - start_t
        if vs.button.count % 2:
            lcd.lcd_display_string_animated_mid('Master', 2, 0.0)
            EASY_MODE = False
        else:
            lcd.lcd_display_string_animated_mid('Beginner', 2, 0.0)
            EASY_MODE = True
    lcd.lcd_clear()
    vs.button.clearCount()

    # Guess Game
    lcd.lcd_display_string_animated_mid('Guess Game', 1, 0.1)
    lcd.lcd_display_string_animated_mid('Get in Position!', 2 , 0.1)

    for player in range(NUM_PLAYER):

        stepper.setMove(0)
        stepper.setMove(-bounds)

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
            if USE_VIDEO:
                if cap.isOpened():
                    ret, bgr_image = cap.read()
            else:
                bgr_image = vs.read()

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
                    lcd.lcd_display_string(' ' * 6 + str(round(score_t, 2)), 2)
                    #before_step = time.time()
                    step = round(-bounds + bounds * 2 * wanted_emotion_prob)
                    stepper.setMove(step)
                    if False:
                        if wanted_emotion_prob > lastProb:
                            stepper.RIGHT_TURN(step)
                        else:
                            stepper.LEFT_TURN(abs(step))
                    #print('[INFO] StepperPos: ' + str(stepper.getPos()) + ' new step: ' + str(step) + ' lastProb ' + str(lastProb))
                    #print('[INFO] Stepper Time in ms ' + str(round(1000*(time.time() - before_step), 2)))

                if predictCount == 0:
                    print('[LOG] Warmup Pipeline Time in ms ' + str((time.time() - last_t)*1000))
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
                    lcd.lcd_display_string(' Player {} Score'.format(player+1),1)
                    lcd.lcd_display_string(' ' * 6 + str(score_t),2)

                lastProb = wanted_emotion_prob
                predictCount += 1

                #print('[LOG] Full Pipeline Time ' + str(time.time() - last_t))
                if predictCount > 1: # ignore warmup
                    pipeline_t.append((time.time() - last_t)*1000)

            elif predictCount > 0:
                # if no face detected
                stepper.setMove(stepper.getPos()-1)

            diff_t = time.time() - start_t
            last_t = time.time()

            loopCount += 1

            if vs.button.count != 0:
                vs.button.clearCount()
                lcd.lcd_clear()
                print('[INFO] Break by Button!')
                break

        if predictCount > 1:
            print('[LOG] PredictionCount ' + str(predictCount))
            print('[LOG] Average Full Pipeline Time in ms ' + str(sum(pipeline_t) / (predictCount-1)))
        pipeline_t.clear()

        # displaying player score
        lcd.lcd_display_string(' ' * 16, 1)
        lcd.lcd_display_string_animated_mid('STOP!', 1, 0.05)
        time.sleep(1)
        scores.append(score_t)
        next_emotion_t = 10
        lcd.lcd_display_string_animated_mid('PL {} score is'.format(player+1), 1, 0.1)
        time.sleep(3)
        if player+1 < NUM_PLAYER:
            lcd.lcd_clear()
            lcd.lcd_display_string_animated_mid('Next Player', 1, 0.1)
            lcd.lcd_display_string_animated_mid('get ready...', 2, 0.1)

    # displaying results
    lcd.lcd_clear()
    time.sleep(0.5)
    lcd.lcd_display_string_animated_mid('Game done!', 1, 0.1)
    #stepper.calibrate()
    stepper.setMove(0)
    time.sleep(2)
    lcd.lcd_display_string_animated_mid('Winner is PL {}'.format(scores.index(max(scores))+1), 1, 0.1)
    lcd.lcd_display_string_animated_mid('with time {}'.format(round(max(scores),2)), 2, 0.1)
    time.sleep(5)

    # Again
    lcd.lcd_clear()
    lcd.lcd_display_string_animated_mid('Next Round in', 1, 0.1)
    start_t, diff_t = time.time(), 0
    while diff_t < 9:
        diff_t = time.time() - start_t
        elapsed_t = ' ' * 7 + str(9-round(diff_t))
        lcd.lcd_display_string(elapsed_t, 2)
        if vs.button.count > 0:
            ENDLESS = False
            break
    lcd.lcd_clear()
    vs.button.clearCount()

# cleanup
lcd.lcd_clear()
lcd.lcd_backlight('off')
vs.button.stop()
stepper.stop()
vs.stop()
if USE_VIDEO:
    cap.release()
cv2.destroyAllWindows()
