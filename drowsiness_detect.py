from scipy.spatial import distance
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import playsound
import imutils
import argparse
from threading import Thread
import time
import dlib
import cv2


#for loading and playing alarm
def sound_alarm():
    	# play an alarm sound
	playsound.playsound('alarm.wav')

#minimum threshold of eye aspect ratio after which alarm will be active
eye_aspect_ratio_threshold = 0.3

#minimum consecutive frames
eye_aspect_ratio_consecutive_frames = 50

#for counting consecutive frames
COUNTER = 0
ALARM_ON = False

#loading face cascade
face_cascade = cv2.CascadeClassifier("haarcascades\haarcascade_frontalface_default.xml")

#calculation of eye_aspect_ratio
def eye_aspect_ratio(eye):
    #distance between two sets of vertical eye landmarks (x,y) co-ordinates
    X = distance.euclidean(eye[1], eye[5])
    Y = distance.euclidean(eye[2], eye[4])
    #distance between horizontal eye landmark
    Z = distance.euclidean(eye[0], eye[3])

    EAR = (X+Y) / (2*Z)
    return EAR

#loading face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#extracting indexes of facial landmark of left and right
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#start webcam
print("Start video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

#time for camera initialization
time.sleep(1)

while True:
    
    #taking the frame
	frame = vs.read()
    #resizing
	frame = imutils.resize(frame, width=450)
    #converting to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	# detect faces in the grayscale frame
	faces = detector(gray, 0)

	# looping through the detected faces
	for rect in faces:
		# determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		#extract left eye coordinates
		leftEye = shape[lstart:lend]
        #extract right eye coordinates
		rightEye = shape[rstart:rend]
        #compute the ratios
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# averaging the eyeaspect ratio
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# checking if the eye aspect ratio is below the blink threshold(eye_aspect_ratio_threshold), 
        # and if so, increment the blink frame counter to get the comparison with eye aspect ratio consecutive frames
		if ear < eye_aspect_ratio_threshold:
			COUNTER += 1
			if COUNTER >= eye_aspect_ratio_consecutive_frames:
				# turn the alarm
				if not ALARM_ON:
					ALARM_ON = True
					#playing alarm
					t = Thread(target=sound_alarm)
					t.deamon = True
					t.start()

				# alart on window
				cv2.putText(frame, "Wake Up!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# if the eye aspect ratio is not below threshold
		else:
			COUNTER = 0
			ALARM_ON = False

		# eye aspect ratio on frame
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# showing frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# press q to exit
	if key == ord("q"):
		break

# close
cv2.destroyAllWindows()
vs.stop()
    



