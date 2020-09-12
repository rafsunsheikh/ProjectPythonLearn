import cv2

faceCascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

while True:
    success,img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Result",img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break


#img = cv2.imread("../Resources/goru1.jpg")
#imgResize = cv2.resize(img,(640,480))
#cv2.imshow("Result",img)
#cv2.waitKey(0)