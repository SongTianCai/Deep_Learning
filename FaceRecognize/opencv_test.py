import numpy as np
import cv2

cap = cv2.VideoCapture(0)

haar=cv2.CascadeClassifier('D:/opencv-3.4/data/haarcascades/haarcascade_frontalface_default.xml')

print(haar)
while(True):

    _, image = cap.read()

    faces = haar.detectMultiScale(image)

    for f_x, f_y, f_w, f_h in faces:
        faces = image[f_y:f_y + f_h, f_x:f_x + f_w]
        faces = cv2.resize(faces, (64, 64))
        test_x = np.array([faces])
        print(faces)
    cv2.imshow('img', faces)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()