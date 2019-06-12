import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/Recogniser.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(Id==1):
            cv2.putText(im,'Prerit', (x,y+h),font, 1,(255,255,255),1)
        elif(Id==2):
            cv2.putText(im,'tony', (x,y+h),font, 1,(255,255,255),1)
        elif(Id!=1):
            cv2.putText(im,'unknown', (x,y+h),font, 1,(0,0,255),1)
        elif(Id!=1):
            cv2.putText(im,'unknown', (x,y+h),font, 1,(0,0,255),1)
        else:
            break;
        
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
