
import cv2
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("../Resources/Save me.mp4")
while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    cv2.imshow("Result", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
         break
