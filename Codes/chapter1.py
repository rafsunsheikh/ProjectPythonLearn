import cv2
img =cv2.imread("../Resources/lena.png")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, imgThresh = cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY)
cv2.imshow("Lena Soderberg",img)
cv2.imshow("Lena Soderberg Thresh",imgThresh)
cv2.waitKey(0)
