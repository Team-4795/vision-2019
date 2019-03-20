import serial
import time
import cv2
import numpy as np
import math
import threading
import sys
from networktables import NetworkTables

print("I exist!")
sys.stdout.flush()

def reject_outliers_2(data, m = 2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return data[s<m]

display = False
distConst = 6973

ser = serial.Serial("/dev/ttyACM0", 9600)

ser.write(b'setcam autoexp 1\r')
ser.write(b'setcam absexp 25\r')
ser.write(b'setcam brightness -3\r')

time.sleep(1)

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
cap.set(5, 19.6)
cap.set(10, -3)
#cap.set(14, 0.15)
#cap.set(15, 0)

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

NetworkTables.initialize(server='10.47.95.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

# Insert your processing code here
print("Connected!")

sys.stdout.flush()

table = NetworkTables.getTable('SmartDashboard')


while True:
    ret, capFrame = cap.read()

    rows,cols,trash = capFrame.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    testFrame = cv2.warpAffine(capFrame,M,(cols,rows))

    #change colorspaces
    hsvFrame = cv2.cvtColor(testFrame, cv2.COLOR_BGR2HSV)
                
    #set bounds for what is "green" and threshold based on that values
    lowerHSVBound = np.array([60, 50, 50])
    upperHSVBound = np.array([100, 255, 255])
    maskFrame = cv2.inRange(hsvFrame, lowerHSVBound, upperHSVBound)

    #perform morphological transformation to remove noise from the image
    #read more here: https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5, 5), np.uint8)
    maskFrame = cv2.morphologyEx(maskFrame, cv2.MORPH_OPEN, kernel)
    
    #get contours of the detected tape
    edgeFrame = cv2.Canny(maskFrame,100,200)
    contour_image, contours, hierarchy = cv2.findContours(edgeFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #iterate over all found contours and find the two largest
    largestSize = -1
    secondLargestSize = -1
    index = 0
        
    tapes = []
    tapeSizes = []
    if len(contours) > 0:
            
            
        #get actual hierarchy from within the bloated datatype
        hierarchy = hierarchy[0]
        #get sizes of all the contours
        for contour in contours:
            #if hierarchy[index, 3] >= 0:
            tapeSizes.append(cv2.contourArea(contour))
            index += 1

        #get rid out of outliers in the data (only keep real tape objects)
        tapeSizes = reject_outliers_2(np.array(tapeSizes), 3).tolist()

        #print(len(tapeSizes))
            
        #add the non-outliers to a new list called tapes
        index = 0
        for contour in contours:
            if cv2.contourArea(contour) in tapeSizes: #this removes contour douplicates caused by the canny edge detector
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
            if M['m00'] != 0:
                x1 = int(M['m10']/M['m00'])
                y1 = int(M['m01']/M['m00'])
            else:
                x1 = 0
                y1 = 0
            if angle < 0 and angle > -40:
                tapeIn.append((x1, y1, cv2.contourArea(tape)))
                cv2.circle(testFrame, (x1, y1), 3, (255, 0, 0), -1)
            if angle > -80 and angle < -40:
                tapeOut.append((x1, y1, cv2.contourArea(tape)))
                cv2.circle(testFrame, (x1, y1), 3, (0, 0, 255), -1)

        #identify the goals in the image using the different types of tapes
        #if an in tape has a nearby out tape to its right side, it means there must be a goal
        for tape in tapeIn:
            smallestDist = 1000000
            pos2 = []
            for tape2 in tapeOut:
                dist = math.sqrt((tape[0] - tape2[0])**2 + (tape[1] - tape2[1])**2)
                xdist = tape[0] - tape2[0]
                if xdist > 0 and dist < smallestDist: #tape is to the right AND is closer than any previous tapes
                    smallestDist = dist
                    pos2 = tape2
            if pos2 == []: #if there is no corresponding tape, dont identify it as a goal
                break
            #pixel coords of the center of the target
            pixel_x = int((tape[0] + pos2[0]) / 2)
            pixel_y = int((tape[1] + pos2[1]) / 2)

            focalLength = 554.2
            
            dist = 305 * (tape[2]**-0.5725)
            targetAngle = math.atan((pixel_x - (np.size(testFrame, 1) / 2)) / focalLength)
            XDist = dist * math.tan(targetAngle)
            
            cv2.circle(testFrame, (pixel_x, pixel_y), 3, (255, 0, 255), -1)
            cv2.putText(testFrame, "Dist: " + str(tape[2]), (pixel_x + 15, pixel_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            if display == False:
                table.putNumber('Z', dist)
                table.putNumber('rawZ', tape[2])
                table.putNumber('X', XDist)
                table.putNumber('Angle', targetAngle)
                #print("Z: " + str(target_depth) + " X: " + str(real_x) + " A: " + str(math.degrees(correction_angle)))
                    
    if display == True:
        #display results
        cv2.imshow("testFrame", testFrame)
        #enabling this can be useful for using a color picker to find exactly what range of HSV
        #works for your image/object
        #cv2.imshow("hsvFrame", hsvFrame)
        cv2.imshow("maskframe", maskFrame)

    key = cv2.waitKey(10)
    if key == 27:
        break


cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
