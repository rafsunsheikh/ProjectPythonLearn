import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("../Resources/staticCamera.mkv")

ret, frame1 =cap.read()
ret, frame2 =cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1,frame2)
    imgGray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
    _, imgThresh = cv2.threshold(imgBlur,20,255,cv2.THRESH_BINARY)
    imgDilated = cv2.dilate(imgThresh,None, iterations = 3)
    contours, _ = cv2.findContours(imgDilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 500:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Status: {}".format('Movement'),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    # cv2.drawContours(frame1,contours,-1,(0,255,0),2)
    cv2.imshow("Video",diff)
    cv2.imshow("Video thresh", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()


    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()

