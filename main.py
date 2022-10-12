import numpy as np
import cv2 as cv
import threading as th

def detector(cap):
    while (True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        left_eye = cv.CascadeClassifier('Weight_AI/left_eye.xml')
        result_left_eye = left_eye.detectMultiScale(gray, scaleFactor=3, minNeighbors=5)
        smile = cv.CascadeClassifier('Weight_AI/smile.xml')
        result_smile = smile.detectMultiScale(gray, scaleFactor=5, minNeighbors=10)
        right_eye = cv.CascadeClassifier('Weight_AI/right_eye.xml')
        result_right_eye = right_eye.detectMultiScale(gray, scaleFactor=3, minNeighbors=2)

        for (x, y, w, h) in result_left_eye:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(frame, "left_eye", (x+w-180 ,y+h+25), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        for (x, y, w, h) in result_smile:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(frame, "smile", (x + w - 180, y + h + 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (x, y, w, h) in result_right_eye:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(frame, "right_eye", (x + w - 180, y + h + 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow('frame', frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

def main():
    cap = cv.VideoCapture(0)
    detector(cap)

if __name__ == '__main__':
    main()
