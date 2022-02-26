
import cv2
import numpy as np
import os 
from training import ids
import sqlite3
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# conn = sqlite3.connect('database.db')
# c = conn.cursor()
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists("saved_model/")
recognizer.read('saved_model/s_model.yml')
# Load prebuilt classifier for Frontal Face detection
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5) #default
    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (200,255,24), 4)
        # Recognize the face belongs to which ID
        Id,con= recognizer.predict(gray[y:y+h,x:x+w])  #Our trained model is working here
        # c.execute("Select name from users where Id=(?);",(Id,))
        # res = c.fetchall()
        # print(res)
        # name = res[0][0]
        name = ids[Id]
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (200,255,24), -1)
        cv2.putText(im, name, (x,y-40), font, 1, (255,255,255), 3)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
