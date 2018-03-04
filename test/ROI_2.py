__author__ = 'ahmaddorri'

import cv2
import numpy as np

img = cv2.imread('images/1.png',0)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(thresh1,kernel,iterations = 1)
#Removing noise from image
blur = cv2.blur(img,(5,5))
#finding edges using edge detection
edges = cv2.Canny(blur, 100 ,200)


laplacian = cv2.Laplacian(edges, cv2.CV_8UC1)
sobely = cv2.Sobel(laplacian,cv2.CV_8UC1, 0, 1, ksize=5)

# Do a dilation and erosion to accentuate the triangle shape
dilated = cv2.dilate(sobely,kernel,iterations = 1)
erosion = cv2.erode(dilated,kernel,iterations = 1)

im2, contours, hierarchy =  cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#keep 10 largest contours
cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    # if our approximated contour has three points, then
    # it must be the road markings
    (x, y, w, h) = cv2.boundingRect(c)
    print((x, y, w, h))
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    if len(approx) == 3:
        screenCnt = approx
        break
# cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Road markings", img)
cv2.waitKey(0)