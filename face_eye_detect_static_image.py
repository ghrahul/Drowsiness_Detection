# importing the necessary library
import cv2 as cv
import numpy as np

#loading cascade classifiers
face_cascade = cv.CascadeClassifier("haarcascades\haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades\haarcascade_eye.xml")

#reading image and convert it into gray scale
img = cv.imread("images/input1.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#detecting faces in image using cascade classifier
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#drawing rectangle and detecting eyes
for (x,y,w,h) in faces:
    cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)

    region_gray = gray[y:y+h, x:x+w]
    region_color = img[y:y+h, x:x+w]

    #detecting eyes
    eyes = eye_cascade.detectMultiScale(region_gray)

    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(region_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

cv.imwrite("output_result/face_eye_detect.jpg",img)
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
