import cv2
import numpy as np

img = cv2.imread('lena.jpg')

numpyEmptyImage = np.zeros([512, 512, 3], np.uint8)

numpyEmptyImage = cv2.circle(numpyEmptyImage, (447, 63), 63, (0, 255, 0), -1)

cv2.imshow('NumpyFrame', numpyEmptyImage)

img = cv2.line(img, (0, 0), (255, 255), (255, 0, 0), 5)

img = cv2.arrowedLine(img, (0, 255), (255, 255), (0, 0, 255), 5)

img = cv2.rectangle(img, (384, 0), (510, 128), (0, 0, 255), 5)

img = cv2.circle(img, (447, 63), 63, (0, 255, 0), -1)

cv2.imshow('Frame', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
