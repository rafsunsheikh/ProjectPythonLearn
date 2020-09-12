import cv2
import numpy as np

img  = cv2.imread("../Resources/lena.png")
kernel = np.ones((5,5),np.uint8)

imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img,(7,7),0)
imgCanny = cv2.Canny(img,150,200)
imgDilation = cv2.dilate(imgCanny,kernel,iterations=1)
imgEroder = cv2.erode(imgDilation,kernel,iterations=1)

cv2.imshow("Normal",img)
cv2.imshow("Grey",imgGrey)
cv2.imshow("Blur",imgBlur)
cv2.imshow("Canny",imgCanny)
cv2.imshow("Dilated",imgDilation)
cv2.imshow("Eroded",imgEroder)
cv2.waitKey(0)