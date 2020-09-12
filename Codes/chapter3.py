import cv2
import numpy as np

img = cv2.imread("../Resources/lambo.png")
print(img.shape)

imgResize = cv2.resize(img,(1000,500))
print(imgResize.shape)

imgCropped = img[0:200,0:300]

cv2.imshow("Image",img)
cv2.imshow("Image Big",imgResize)
cv2.imshow("Image Cropped",imgCropped)
cv2.waitKey(0)