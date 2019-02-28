# importing necessary library
import cv2 as cv
import numpy as np

#loading cascading class
face_cascade = cv.CascadeClassifier("haarcascades\haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades\haarcascade_eye.xml")

#capturing video through webcam
video_capture = cv.VideoCapture(0)

# reading all frames through webcam
while True:
    ret, frame = video_capture.read()
    #video feed should be flipped so that it appears mirror like
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        region_gray = gray[y:y+h, x:x+w]
        region_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(region_gray)

        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(region_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)

    cv.imshow('Video', frame)
    if(cv.waitKey(1) & 0xFF == ord('q')):
        break

#release video capture
video_capture.release()
cv.waitKey(0)
cv.destroyAllWindows()
