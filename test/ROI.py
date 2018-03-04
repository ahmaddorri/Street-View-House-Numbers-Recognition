# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np


# load the example image
image = cv2.imread("images/2.png")
# image = cv2.imread("images/21.png")
# image = cv2.imread("images/19.png")
# image = cv2.imread("images/25.png")
# image = cv2.imread("images/4.png")
# image = cv2.imread("images/5.png")
# image = cv2.imread("images/00.png")

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(blurred, 50, 200, 255)
cv2.imshow('thresh', image)
cv2.waitKey(0)
cv2.imshow('gray', gray)
cv2.waitKey(0)
# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(blurred, 0, 255,cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]
cv2.imshow('thresh', thresh)
cv2.waitKey(0)


cv2.adaptiveThreshold(dst=thresh,src=thresh, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=499,C=-1)[1]
# thresh = cv2.threshold(blurred, thresh=180, maxval=255, type=cv2.THRESH_BINARY )[1]
cv2.imshow('thresh', thresh)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)
thresh=255-thresh
cv2.imshow('thresh', thresh)
cv2.waitKey(0)




# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
digitCnts = []
# print(cnts)
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    print((x, y, w, h))
    cv2.rectangle(image,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),2)

    # if the contour is sufficiently large, it must be a digit
    if w >= 90 and w<=200 and (h >= 150 and h <= 400):
        digitCnts.append(c)

for c in digitCnts:
    (x, y, w, h) = cv2.boundingRect(c)
    # cv2.rectangle(image,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),2)

cv2.imshow('image', image)
cv2.waitKey(0)

