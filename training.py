
import cv2
import os
import numpy as np
from PIL import Image

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

ids = {}

def getImagesAndLabels(path):

    # Getting all file paths
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    ID = []
    for imagePath in imagePaths:

        # converting image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        # converting PIL image to numpy array using array() method of numpy
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        name = os.path.split(imagePath)[-1].split(".")[2]
        # print(id)
        # print(name)
        # Getting the face from the training images
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            # Add the ID to IDs
            ID.append(id)
            ids[id] = name
    # Passing the face array and IDs array
    return faceSamples,ID

# Getting the faces and IDs
faces,idss = getImagesAndLabels('training_data')
print(ids)

# Training the model using the faces and IDs
recognizer.train(faces, np.array(idss))

# Saving the model into s_model.yml
assure_path_exists('saved_model/')
recognizer.write('saved_model/s_model.yml')
