import numpy as np
import cv2
import math

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
	
#read test image
testFrame = cv2.imread("test-images/CargoStraightDark48in.jpg");

#change colorspaces
hsvFrame = cv2.cvtColor(testFrame, cv2.COLOR_BGR2HSV)

#set bounds for what is "green" and threshold based on that values
lowerHSVBound = np.array([50, 100, 100])
upperHSVBound = np.array([84, 255, 255])
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

tapes = []
tapeSizes = []

#get actual hierarchy from within the bloated datatype
hierarchy = hierarchy[0]

#get sizes of all the contours
for contour in contours:
	if hierarchy[index, 3] >= 0:
		tapeSizes.append(cv2.contourArea(contour))
	index += 1

#get rid out of outliers in the data (only keep real tape objects)
tapeSizes = reject_outliers(np.array(tapeSizes)).tolist()

#add the non-outliers to a new list called tapes
index = 0
for contour in contours:
	if hierarchy[index, 3] >= 0 and cv2.contourArea(contour) in tapeSizes: #this removes contour douplicates caused by the canny edge detector
		tapes.append(contour)
	index += 1


#get metadata of the tape objects which is all we need (position and angle) and seperate them into two groups:
#tape that is slanted inwards, and tape that is slanted outwords
tapeIn = []
tapeOut = []

for tape in tapes:
	box = cv2.minAreaRect(tape)
	angle = box[2]
	M = cv2.moments(tape)
	x1 = int(M['m10']/M['m00'])
	y1 = int(M['m01']/M['m00'])
	if angle < 0 and angle > -20:
		tapeIn.append((x1, y1))
		#cv2.circle(testFrame, (x1, y1), 3, (255, 0, 0), -1)
	if angle > -80 and angle < -50:
		tapeOut.append((x1, y1))
		#cv2.circle(testFrame, (x1, y1), 3, (0, 0, 255), -1)

#identify the goals in the image using the different types of tapes
#if an in tape has a nearby out tape to its right side, it means there must be a goal
for tape in tapeIn:
	smallestDist = 1000000
	pos2 = []
	for tape2 in tapeOut:
		dist = tape[0] - tape2[0]
		if dist > 0 and dist < smallestDist: #tape is to the right AND is closer than any previous tapes
			smallestDist = dist
			pos2 = tape2
	if pos2 == []: #if there is no corresponding tape, dont identify it as a goal
		break
	cv2.circle(testFrame, (int((tape[0] + pos2[0]) / 2), int((tape[1] + pos2[1]) / 2)), 3, (255, 0, 255), -1)


#display results
cv2.imshow("testFrame", testFrame)
#enabling this can be useful for using a color picker to find exactly what range of HSV
#works for your image/object
#cv2.imshow("hsvFrame", hsvFrame)
#cv2.imshow("maskFrame", maskFrame)

cv2.waitKey()

# When everything done, release the capture
cv2.destroyAllWindows()
