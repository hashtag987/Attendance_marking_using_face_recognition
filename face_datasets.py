import cv2
import os
import random
import sqlite3
#from training import ids
#conn = sqlite3.connect('database.db')
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Starting the web cam by invoking the VideoCapture method
vid_cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set unique id for each individual person
face_id = random.randrange(101,999,1)
count = 0
assure_path_exists("training_data/")
uname = input("Enter ur name: ")

# c = conn.cursor()
# c.execute("INSERT INTO users (name) VALUES (?);",(uname,))
# face_id = c.lastrowid

while(True):
    _, image_frame = vid_cam.read()
    # Converting each frame to grayscale image
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    # Detecting different faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # Looping through all the detected faces in the frame
    for (x,y,w,h) in faces:
        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("training_data/Person." + str(face_id) + '.' + uname + "."+ str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', image_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count>50:
        break
vid_cam.release()
cv2.destroyAllWindows()
