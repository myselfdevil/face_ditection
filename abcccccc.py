import cv2 #open cv
import numpy as  np
import time
face_cascade = cv2.CascadeClassifier('C:\\aaaaaaaa\\haarcascades\\haarcascade_frontalface_default.xml')#loading the face xml files

eye_cascade = cv2.CascadeClassifier('C:\\aaaaaaaa\\haarcascades\\haarcascade_eye.xml')
 #loading the eyes xml files
r = False
#img=cv2.imread('C:\\aaaaaaaa\\haarcascades\\ab.jpg')
cap=cv2.VideoCapture(0)
t1 = time.time();
while True:
    ret,img = cap.read()
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY ) #converting to grayscale
    facial = face_cascade.detectMultiScale(gray, 1.3, 5)
    #converting to grayscale
    t2 = time.time();
    if r == False:
      dt1 = t2-t1
      print(dt1)
      r = True
    for (x,y,w,h) in facial:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #draw blue rectangle around eyes and lips
        roi_gray = gray[y:y+h, x:x+w] #locations of the face in grayscale
        roi_color = img[y:y+h, x:x+w] #locations of converted grayscale
        eyes = eye_cascade.detectMultiScale(roi_gray)# detect eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0),2) # drawing a blue rectangle around the eyes

    cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:          
      break     

cap.release()  
cv2.destroyAllWindows()
