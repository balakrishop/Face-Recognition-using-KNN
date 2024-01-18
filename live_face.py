import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

with open('data/faces.pkl', 'rb') as w:
    faces = pickle.load(w)

with open('data/names.pkl', 'rb') as f:
    labels = pickle.load(f)

facec = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

print('Shape of Faces matrix --> ', faces.shape)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces,labels)

# Face Recognition using KNN
while True:
    ret, fr = cam.read()
    if ret == True:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        face_coordinates = facec.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_coordinates:
            fc = fr[y:y + h, x:x + w, :]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1,-1)
            text = knn.predict(r)
            cv2.putText(fr, text[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + w), (0, 0, 255), 2)

        cv2.imshow('livetime face recognition', fr)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("error")
        break

cv2.destroyAllWindows()
