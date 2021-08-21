import cv2
import numpy as np


def click_events(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_COMPLEX
        print(x, y)
        text = 'X : ' + str(x) + ' Y : ' + str(y)
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_COMPLEX
        print(blue, green, red)
        text = 'B : ' + str(blue) + ' G : ' + str(green) + ' R : ' + str(red)
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


img = cv2.imread('lena.jpg')
#img = np.zeros((512, 512, 3), np.uint8)
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_events)
cv2.waitKey(0)
cv2.destroyAllWindows()
