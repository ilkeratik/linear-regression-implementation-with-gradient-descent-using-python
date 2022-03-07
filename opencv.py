import cv2
import numpy as np
img = np.zeros((10,10,3), np.uint8)
img[0,:] = (255,0,0) 
img[1,:] = (0,0,255)
img[2,:] = (255,255,190)
img[3,:] = (255,255,160)
img[4,:] = (255,255,130)
img[5,:] = (255,255,100)
cv2.imshow('sad',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
