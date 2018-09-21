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
#from imutils.video.pivideostream import PiVideoStream
import modules.PiVideoStream as PiVideoStream
import modules.lcddriver as lcddriver
import modules.Stepper as Stepper
import time
import RPi.GPIO as GPIO
from random import shuffle

TUTORIAL = False
NUM_PLAYER = 2
ROUND_TIME = 30 # seconds
EMOTIONS = ["neutral", "happy", "sad", "surprise", "angry"]
next_emotion = 10
scores = []

USE_PICAM = True # If false, loads video file source
USE_THREAD = True

# initialize displayed
lcd = lcddriver.lcd()

# set up stepper
bounds = 30
stepper = Stepper.Stepper(bounds)
step = 1 # degree
step_scale = (bounds * 2) / 100

# hyper-parameters for bounding box shape
emotion_offsets = (20, 40)

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Select video or webcam feed
cap = None
if USE_PICAM:
    if USE_THREAD:
        vs = PiVideoStream.PiVideoStream().start()
        time.sleep(1)
    else:
        # initialize the camera and stream
        camera = PiCamera()
        camera.resolution = (320, 240)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(320, 240))
        cap = camera.capture_continuous(rawCapture, format="bgr",
        	use_video_port=True)
        time.sleep(1)
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

lcd.lcd_display_string_animated_mid('EMOTRONOM', 1, 0.2)
time.sleep(1)
lcd.lcd_display_string_animated('Emotion Detector', 2, 0.1)
time.sleep(3)
lcd.lcd_clear()

if TUTORIAL:
    # TODO: text verbinden, steppa raus
    lcd.lcd_display_string_animated_mid('TUTORIAL', 1, 0.1)
    time.sleep(0.5)
    lcd.lcd_display_string_long('There are five emotions. Neutral, Happy, Angry, Sad and Surprise.', 2, 0.15)
    time.sleep(0.1)
    lcd.lcd_display_string_long('Guess the emotion by facial expressions. Then to keep the amplitude high.', 2, 0.15)
    time.sleep(0.1)
    lcd.lcd_display_string_long_2('Who perseveres the longest, is the KING.', 2, 0.15)
    time.sleep(0.1)
    lcd.lcd_clear()
    lcd.lcd_display_string_animated('    Have Fun    ', 2, 0.1)
    time.sleep(1)
    lcd.lcd_clear()

lcd.lcd_display_string_animated_mid('First Round', 1, 0.1)
time.sleep(3)

for player in range(NUM_PLAYER):

    stepper.calibrate()
    stepper.LEFT_TURN(bounds)
    mixed_emotions = EMOTIONS.copy()
    shuffle(mixed_emotions)
    searched_emotion = mixed_emotions.pop()

    lcd.lcd_clear()
    lcd.lcd_display_string_animated_mid('Player {}'.format(player), 1, 0.1)
    start_t, diff_t = time.time(), 0
    while diff_t < 10:
        diff_t = time.time() - start_t
        elapsed_t = ' ' * 7 + str(5-round(diff_t))
        lcd.lcd_display_string(elapsed_t, 2)
    lcd.lcd_clear()

    lcd.lcd_display_string('Player {}:'.format(player),1)
    start_t = last_t = time.time()
    score_t = diff_t = 0
    lcd.lcd_display_string(str(score_t),2)

    while diff_t < ROUND_TIME:

        if round(diff_t) > next_emotion:
            next_emotion += 10
            searched_emotion = mixed_emotions.pop()
            print('new searched Emotion: ' + searched_emotion)

        # get image from picamera
        if USE_THREAD:
            bgr_image = vs.read()
        else:
            ret, bgr_image = cap.read()
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # detect faces
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
    			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # select dominating face
        curr, major = 0, None
        for i, (x, y, w, h) in enumerate(faces):
            if w * h > curr:
                curr = w * h
                major = i

        # predict emotions
        if major != None:
            print('start face prediction...')
            face_coordinates = faces[major]
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
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            print('emotion_text: ' + emotion_text)

            if emotion_text == searched_emotion:
                # TODO: need to reach specific probability, before time is added
                score_t  = score_t + (time.time() - last_t)
                lcd.lcd_display_string(str(round(score_t, 2)), 2)
                step = step_scale * emotion_probability - stepper.getPos()
                stepper.RIGHT_TURN(step)
            else:
                lcd.lcd_display_string(str(round(score_t, 2)), 2)
                step = step_scale * emotion_probability - stepper.getPos()
                stepper.LEFT_TURN(step)

            if emotion_text == 'disgust' or emotion_text == 'fear':
                print(emotion_text + ' detected, so more emotions present')

        diff_t = time.time() - start_t
        last_t = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    lcd.lcd_display_string_animated_mid('STOP!', 1, 0.05)
    time.sleep(1)
    scores.append(score_t)
    next_emotion = 10
    lcd.lcd_display_string_animated_mid('PL {} score is'.format(player), 1, 0.1)
    time.sleep(5)

lcd.lcd_clear()
time.sleep(0.5)
lcd.lcd_display_string_animated_mid('Game is done.', 1, 0.1)
time.sleep(2)
lcd.lcd_display_string_animated_mid('Winner is PL {}'.format(scores.index(max(scores))), 1, 0.1)
lcd.lcd_display_string_animated_mid('with time {}'.format(round(max(scores),2)), 2, 0.1)
time.sleep(10)
lcd.lcd_clear()
lcd.lcd_backlight('off')

if USE_THREAD:
    vs.stop()
else:
    cap.release()
    rawCapture.close()
    camera.close()

cv2.destroyAllWindows()
