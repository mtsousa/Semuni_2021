# Author: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
# Subject: Princípios de Visão Computacional
#
# This code detect faces from the ORL Database of faces and
# try to recognize the faces from a block test or at real time
# from webcam. From ORL Database i have used the first fourteen 
# image sets and added the last set with my own images.
#
# The images for training set can be download here: https://www.kaggle.com/kasikrit/att-database-of-faces
# This project is based on the tutorial from SuperDataScience
# which can be accessed by https://www.superdatascience.com/blogs/opencv-face-recognition

import cv2
import numpy as np

# Function to detect faces
def detect_face(img):
    
    # Load OpenCV Haar Cascade
    face_cascade = cv2.CascadeClassifier('data/classifier/haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    
    # If no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    img_faces = []
    # Extract the face area of each face detected
    for i in range(len(faces)):
        aux = img
        (x, y, w, h) = faces[i]
        aux = aux[y:y+w, x:x+h]
        img_faces.append(aux)
    
    #return only the face part of the image
    return img_faces, faces

# Function to draw the rectangle on the detected face
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to put the label on the detected face
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

# Function to prepare the training data
# Read the training test and organize the datas
def prepare_training_data():
    
    file = open('data/training_file.txt', 'r')
    lines = file.readlines()

    #lists to hold all subject faces, labels and found faces
    faces = []
    labels = []
    face_found = [] 

    for i in range(len(lines)):
        aux = lines[i]
        aux = aux.split(';')

        dir = aux[0]
        label = int(aux[1])

        new_dir = 'data/' + dir

        # Read image
        image = cv2.imread(new_dir, cv2.IMREAD_GRAYSCALE)
        
        # Display an image window to show the image 
        cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
        cv2.waitKey(100)
        
        # Detect face
        face_found, rect = detect_face(image)
        
        # Add all detected faces
        if face_found is not None:
            for m in range(len(face_found)):
                aux = face_found[m]
                # Add face to list of faces
                faces.append(aux)
                # Add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    file.close()
    
    return faces, labels

# Function to recognize a person in an passed image
def recognize_img(color, gray):

    # Make a copy of the image
    img = gray.copy()
    orig = color.copy()

    # Detect face from the image
    face, rect = detect_face(img)
    if face is not None:
        for k in range(len(face)):
            aux = face[k]
            
            # Recognize the image using the trained face recognizer 
            label, confidence = face_recognizer.predict(aux)
            
            # Get name or the label returned
            if label == 9:
                label_text = 'Matheus'
            else:
                label_text = str(label)

            # Draw a rectangle around face detected
            draw_rectangle(orig, rect[k])
            
            # Put the name of detected person
            draw_text(orig, label_text, rect[k][0], rect[k][1]-5)

    return orig 

# Load the training data
print("Preparing data for training...")
faces, labels = prepare_training_data()

# Create the LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Training the face recognizer
face_recognizer.train(faces, np.array(labels))

# Define the type of evaluation
cont = input('Define how you want to evalute the code, test block (t) or real time (r): ')
while cont != 't' and cont != 'r':
    print('\nThe answer must be \'t\' or \'r\'...')
    cont = input('Define how you want to evalute the code, for test block (t) or real time (r): ')

# Try to recognize faces from a test block
# created with images from outside of training_file
if cont == 't':
    print("Loading images...")
    file = open('data/test_file.txt')
    line = file.readlines()
    
    for i in range(len(line)):
        aux = line[i].split(';')
        img = cv2.imread('data/' + aux[0], cv2.IMREAD_GRAYSCALE)
        

        if img is not None:
            face_img = recognize_img(img, img)
            cv2.imshow('img', cv2.resize(face_img, (400, 500)))
            cv2.waitKey(0)

    cv2.destroyAllWindows()    
    file.close()

# Try to recognize the face from webcam frames
else:
    print("Capturing video...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while (True):
        res, img = cap.read()
        if res == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = recognize_img(img, gray)
        
            cv2.imshow('video_capture', cv2.resize(faces, (500, 400)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()