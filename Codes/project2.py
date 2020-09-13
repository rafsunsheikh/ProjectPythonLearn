import cv2
import numpy as np

widthImg = 640
heightImg = 480


cap = cv2.VideoCapture(0)
cap.set(3,widthImg)
cap.set(4,heightImg)
cap.set(10,150)

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel =  np.ones((5,5))
    imgDilated = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThreshold = cv2.erode(imgDilated,kernel,iterations=1)

    return imgThreshold

def getContours(img):
    contours, Hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            print(len(approx))
            objCorners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

while True:
    success,img = cap.read()
    img = cv2.resize(img,(widthImg,heightImg))
    imgContours = img.copy()
    imgThreshold = preProcessing(img)
    cv2.imshow("Result",imgContours)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break