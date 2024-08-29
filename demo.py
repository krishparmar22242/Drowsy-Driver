from scipy.spatial import distance  # To calculate the euclidean distance
from imutils import face_utils # To get landmarks of eyes and mouths
from pygame import mixer  #To play the alarm
import numpy as np
import imutils
import dlib
import cv2

mixer.init()            #Initializing mixer module
mixer.music.load("alarm.mp3")
def eye_aspect_ratio(eye):

    sdistance1=distance.euclidean(eye[1],eye[5])
    sdistance2 = distance.euclidean(eye[2], eye[4])
    ldistance=distance.euclidean(eye[0],eye[3])
    eye_aspect_ratio=(sdistance1+sdistance2)/(2.0 * ldistance)
    return eye_aspect_ratio

def mouth_aspect_ratio(shape):
    top_lip=shape[50:53]
    top_lip_entire=np.concatenate((top_lip,shape[61:64]))

    bottom_lip=shape[56:59]
    bottom_lip_entire=np.concatenate((bottom_lip,shape[65:68]))

    top_lip_mean=np.mean(top_lip_entire,axis=0)
    bottom_lip_mean=np.mean(bottom_lip_entire,axis=0)
    MAR=abs(top_lip_mean[1]-bottom_lip_mean[1])
    return MAR

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#are used to define the start and end indices of the facial landmarks corresponding to the left and right
#eyes. The face_utils.FACIAL_LANDMARKS_68_IDXS dictionary contains predefined indices for the 68 facial
#landmarks detected by the dlib library.

(lstart,lend)=face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rstart,rend)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
cap1=cv2.VideoCapture(0)

yawn_thresh=26
threshold_value=0.25
eye_blink=0
frame_count=15
while cap1:
    ret,frame=cap1.read()
    gray_face=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray_face)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        rectangle = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        landmarks=predictor(gray_face,face)
        landmarks_array=face_utils.shape_to_np(landmarks)
        lefteyelandmarks=landmarks_array[lstart:lend]       # Extract Left eye landmarks
        righteyelandmarks = landmarks_array[rstart:rend]        # Extract right eye landmarks
        leftear=eye_aspect_ratio(lefteyelandmarks)      #Calculate idividual eye ear
        rightear = eye_aspect_ratio(righteyelandmarks)       #Calculate idividual eye ear
        avgear=(leftear+rightear)/2.0   #calculate average ear
	MAR = mouth_aspect_ratio(landmarks_array)
        leftEyeHull = cv2.convexHull(lefteyelandmarks)		#Covex hull is the minimum boundary that completly enclose or cover the object
        rightEyeHull = cv2.convexHull(righteyelandmarks)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) 	# Contour is a line that joins all the convex hull points 
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
        if avgear<threshold_value:
            eye_blink+=1
            print(eye_blink)
            if eye_blink>=frame_count:
                cv2.putText(frame,"==========Alert==========",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),1)
                mixer.music.play()
        else:
            eye_blink=0
	if MAR > yawn_thresh:
                    cv2.putText(frame, "************YAWN ALERT!************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "************YAWN ALERT!************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
    cv2.imshow("Frame",frame)
    if cv2.waitKey(25) & 0xff==ord('q'):
        break
cap1.release()
cv2.destroyAllWindows()