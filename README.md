# Drowsy-Driver
In an era where transportation is an integral part of our daily lives, ensuring road safety has become a critical concern. According to the reports of the National Crime Records Bureau (NCRB) about 135,000 road accidents-related deaths occur every year in India.
In this project Drowsiness Detection System that utilizes a video or webcam for monitoring drivers without the need for complex and uncomfortable equipment. Implement a real-time video analysis system that processes each frame efficiently. This will enable the continuous monitoring of the driver's facial expressions, allowing for immediate detection of drowsiness onset. Our aim is to implement a facial feature localization algorithm that can precisely identify the location of eyes and mouth in video frames, overcoming challenges posed by varying lighting conditions and driver head movements. Employ machine learning techniques to analyze the localized facial features and detect signs of drowsiness. Train the system on a diverse dataset to enhance accuracy and adaptability to individual drivers. Design a user-friendly interface for the Drowsiness Detection System, ensuring ease of integration with existing in-car technologies. The system should provide clear alerts and warnings to the driver without causing distraction.

Features

Real-time detection: Continuously monitors facial features to detect drowsiness.

Eye aspect ratio (EAR): Measures the eye closure to determine drowsiness.

Mouth aspect ratio (MAR): Measures mouth opening to detect yawning.

Alert system: Plays an alarm sound when drowsiness or yawning is detected.

Requirements

Python 3.x
OpenCV (cv2)
dlib
numpy
imutils
pygame
shape_predictor_68_face_landmarks.dat  download from here      https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat

Code Overview

Import Libraries

scipy.spatial.distance: Used to calculate the Euclidean distance between facial landmarks.
imutils.face_utils: Extracts facial landmarks (eyes and mouth) from detected faces.
pygame.mixer: Used to play the alarm sound.
OpenCV (cv2): Handles video capture and image processing.
dlib: Detects faces and facial landmarks.

Main Functions
eye_aspect_ratio(eye): Calculates the eye aspect ratio (EAR) to measure eye closure.
mouth_aspect_ratio(shape): Calculates the mouth aspect ratio (MAR) to detect yawning.

Detection Logic
Eye Blink Detection: Monitors the EAR to detect prolonged eye closure, indicating drowsiness.

Yawn Detection: Monitors the MAR to detect when the mouth is open wide enough to indicate yawning.
Alarm Activation: When drowsiness or yawning is detected, an alarm sound is triggered using the pygame.mixer.

