import numpy as np
import cv2

#read test image
testFrame = cv2.imread("test-images/1ftH7ftD0Angle0Brightness.jpg");

#change colorspaces
hsvFrame = cv2.cvtColor(testFrame, cv2.COLOR_BGR2HSV)

#set bounds for what is "green" and threshold based on that values
lowerHSVBound = np.array([85, 100, 100])
upperHSVBound = np.array([100, 255, 255])
maskFrame = cv2.inRange(hsvFrame, lowerHSVBound, upperHSVBound)

#perform morphological transformation to remove noise from the image
#read more here: https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
kernel = np.ones((5,5),np.uint8)
maskFrame = cv2.morphologyEx(maskFrame, cv2.MORPH_OPEN, kernel)

#display results
cv2.imshow("testFrame", testFrame)
#enabling this can be useful for using a color picker to find exactly what range of HSV
#works for your image/object
#cv2.imshow("hsvFrame", hsvFrame)
cv2.imshow("maskFrame", maskFrame)

cv2.waitKey()

# When everything done, release the capture
cv2.destroyAllWindows()
