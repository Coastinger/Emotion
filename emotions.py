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

tutorial = True

USE_PICAM = True # If false, loads video file source
USE_THREAD = True

# initialize displayed
lcd = lcddriver.lcd()

# set up stepper
bounds = 30
stepper = Stepper.Stepper(bounds)
step = 5 # degree

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

#cv2.namedWindow('window_frame')

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

lcd.lcd_display_string_animated('   EMOTRONOM    ', 1, 0.2)
time.sleep(1)
lcd.lcd_display_string_animated('Emotion Detector', 2, 0.1)
time.sleep(3)
lcd.lcd_clear()

if tutorial:
    lcd.lcd_display_string_animated('    TUTORIAL    ', 1, 0.1)
    time.sleep(0.5)
    lcd.lcd_clear()
    lcd.lcd_display_string_long('There are five emotions.', 1, 0.28)
    time.sleep(0.1)
    lcd.lcd_display_string_long('Neutral, Happy, Angry, Sad and Surprise.', 2, 0.28)
    time.sleep(0.1)
    lcd.lcd_display_string_long('Guess the emotion by facial expressions.', 1, 0.28)
    time.sleep(0.1)
    lcd.lcd_display_string_long('Then to keep the amplitude high.', 2, 0.28)
    time.sleep(0.1)
    lcd.lcd_display_string_long('Who perseveres the longest, is the KING.', 1, 0.28)
    lcd.lcd_clear()
    lcd.lcd_display_string_animated('    Have Fun    ', 2, 0.1)
    time.sleep(1)
    lcd.lcd_clear()

lcd.lcd_display_string_animated('First Round', 1, 0.1)
lcd.lcd_clear()
lcd.lcd_display_string_animated('Countdown Pl 1', 1, 0.1)
start = time.time()
diff = 0
while diff < 5:
    diff = time.time() - start
    elapsed = '       ' + str(5-round(diff))
    lcd.lcd_display_string(elapsed, 2)
lcd.lcd_clear()
lcd.lcd_display_string('Player 1:',1)

while True: #cap.isOpened():
    if USE_THREAD:
        bgr_image = vs.read()
    else:
        ret, bgr_image = cap.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # select dominating face
    curr, major = 0, None
    for i, (x, y, w, h) in enumerate(faces):
        if w * h > curr:
            curr = w * h
            major = i

    #for face_coordinates in faces:
    if major != None:
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
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            angry_str = 'Angry:  ' + str(round(emotion_probability,2))
            lcd.lcd_display_string(angry_str , 1)
            stepper.RIGHT_TURN(step)
            #color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            sad_str = 'Sad:  ' + str(round(emotion_probability,2))
            lcd.lcd_display_string(sad_str , 2)
            stepper.LEFT_TURN(step)
            #color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            stepper.LEFT_TURN(0)
            #color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            stepper.RIGHT_TURN(0)
            #color = emotion_probability * np.asarray((0, 255, 255))
        else:
            lcd.lcd_display_string('--- Neutral ---', 2)
            #color = emotion_probability * np.asarray((0, 255, 0))

        #color = color.astype(int)
        #color = color.tolist()

        #draw_bounding_box(face_coordinates, rgb_image, color)
        #draw_text(face_coordinates, rgb_image, emotion_mode,
                  #color, 0, -45, 1, 1)

    #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #cv2.imshow('window_frame', bgr_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if USE_THREAD:
    vs.stop()
else:
    cap.release()
    rawCapture.close()
    camera.close()

cv2.destroyAllWindows()
