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

#get contours of the detected tape
edgeFrame = cv2.Canny(maskFrame,100,200)
contourFrame, contours, hierarchy = cv2.findContours(edgeFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#iterate over all found contours and find the two largest
largestSize = -1
secondLargestSize = -1
index = 0
tape1 = []
tape2 = []

#get actual hierarchy from within the bloated datatype
hierarchy = hierarchy[0]

for contour in contours:
	if hierarchy[index, 3] >= 0: #this removes contour douplicates caused by the canny edge detector
		size = cv2.contourArea(contour)
		if size > largestSize:
			tape1 = contour
			largestSize = size
	index += 1
	
index = 0
for contour in contours:
	if hierarchy[index, 3] >= 0:
		size = cv2.contourArea(contour)
		if size > secondLargestSize and size < largestSize:
			tape2 = contour
			secondLargestSize = size
	index += 1

#find the position of the center of these contours and display them on the original frame
M1 = cv2.moments(tape1)
M2 = cv2.moments(tape2)

x1 = int(M1['m10']/M1['m00'])
y1 = int(M1['m01']/M1['m00'])

x2 = int(M2['m10']/M2['m00'])
y2 = int(M2['m01']/M2['m00'])

cv2.circle(testFrame, (x1, y1), 3, (255, 0, 0), -1)
cv2.circle(testFrame, (x2, y2), 3, (255, 0, 0), -1)

#display results
cv2.imshow("testFrame", testFrame)
#enabling this can be useful for using a color picker to find exactly what range of HSV
#works for your image/object
#cv2.imshow("hsvFrame", hsvFrame)
cv2.imshow("maskFrame", maskFrame)

cv2.waitKey()

# When everything done, release the capture
cv2.destroyAllWindows()
