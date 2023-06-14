import cv2
import numpy as np

img=np.zeros((512,512,3),np.int8)

while True:
    cv2.imshow('image',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.waitKey(1)