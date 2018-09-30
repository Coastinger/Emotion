# Emotion Metronom

The Emotion Metronom is a Mini-Game-Modul to play with your facial expressions. Its software recognizes human faces and their corresponding emotions from webcam feed. Powered by OpenCV and Deep Learning deployed on an Raspberry Pi system.

![Demo](https://github.com/Coastinger/Emotronom/tree/master/demo/Emotronom_Demo.gif)

## Deep Learning Model

The model used is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf) written by Octavio Arriaga, Paul G. Plöger, and Matias Valdenegro.

![Model](https://i.imgur.com/vr9yDaF.png?1)

## Circuit

![Circuit](https://github.com/Coastinger/Emotronom/tree/master/demo/Emotronom_Circuit.jpg?1)

Note: As power there are two batteries displayed. In fact I used 4 batteries from the brand eneloop. The brand is important if u do not want to use a voltage regulator, cause eneloop gives exactly 5V +/- 0.3.

## Blueprint

![Blueprint](https://github.com/Coastinger/Emotronom/tree/master/demo/Emotronom_Blueprint_800x600.jpg?1)

## Credit

* Computer vision powered by OpenCV.
* Neural network scaffolding powered by Keras with Tensorflow.
* Convolutional Neural Network (CNN) deep learning architecture is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf).
* Pretrained Keras model and much of the OpenCV code provided by GitHub user [oarriaga](https://github.com/oarriaga).

## Useful Links

* Object Detection with Pi [tutorial](https://www.youtube.com/watch?v=npZ-8Nj1YwY&index=42&list=WL&t=938s).
* Stepper Motor Tutorial [tutorial](https://www.youtube.com/watch?v=4fHL6BpJrC4).
* LCD Display Tutorial [turorial](https://www.youtube.com/watch?v=B0AQDOTUq2M&t=326s).
* PiCamera and OpenCV [tutorial](https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/).
