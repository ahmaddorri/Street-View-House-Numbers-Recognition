# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np


# load the example image
image = cv2.imread("images/1.png")
print(image.shape)
# pre-process the image by resizing it, converting it to
#graycale, blurring it, and computing an edge map
image = imutils.resize(image,height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

sobel_horizontal = cv2.Sobel(blurred,cv2.CV_8UC1,1,0,ksize=5)
sobel_vertical = cv2.Sobel(blurred,cv2.CV_8UC1,0,1,ksize=5)

blurred = cv2.GaussianBlur(sobel_horizontal, (5, 5), 0)

cv2.imshow('Sobel Horizontal Filter',sobel_horizontal)
# cv2.imshow('Sobel Vertical Filter',sobel_vertical)
cv2.waitKey(0)



# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(blurred, 90, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)
thresh=255-thresh
cv2.imshow('thresh', thresh)
cv2.waitKey(0)


cv2.adaptiveThreshold(dst=thresh,src=thresh, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=3,C=0)[1]
cv2.imshow('thresh', thresh)
cv2.waitKey(0)

# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []
# print(cnts)
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    print((x, y, w, h))
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    # if the contour is sufficiently large, it must be a digit
    if w >= 9 and w<=35 and (h >= 10 and h <= 50):
        digitCnts.append(c)


for c in digitCnts:
    (x, y, w, h) = cv2.boundingRect(c)
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('image', image)
cv2.waitKey(0)