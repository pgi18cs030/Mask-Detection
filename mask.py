from keras.models import load_model
import cv2
import numpy as np
import tkinter

root = tkinter.Tk()
root.withdraw()

model = load_model('C:/Users/hp/Desktop/new mask/final.h5')

face_det_classifier = cv2.CascadeClassifier('workspace.xml')

vid_source = cv2.VideoCapture(0)

text_dict = {0: 'Mask ON', 1: 'No Mask'}
rect_color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

while (True):

    ret, img = vid_source.read()
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_det_classifier.detectMultiScale(grayscale_img, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = grayscale_img[y:y + w, x:x + w]
        resized_img = cv2.resize(face_img, (128, 128))
        normalized_img = resized_img / 255.0
        #reshaped_img = np.reshape(normalized_img, (128, 128, 3))
        result = model.predict(normalized_img)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), rect_color_dict[label], -1)
        cv2.putText(img, text_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        break
    cv2.imshow('LIVEFACEMASK', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
vid_source.release()