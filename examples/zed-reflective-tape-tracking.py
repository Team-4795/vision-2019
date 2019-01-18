import sys
import pyzed.sl as sl
import numpy as np
import cv2
import math


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def reject_outliers_2(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return data[s<m]

zed = sl.Camera()

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
init.camera_fps = 30
init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE
init.coordinate_units = sl.UNIT.UNIT_METER

# Open the camera
err = zed.open(init)

if err != sl.ERROR_CODE.SUCCESS:
    print(repr(err))
    zed.close()
    exit(1)

zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, 20, False)
zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS, 0, False)
zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST, 0, False)

runtime = sl.RuntimeParameters()
runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

# Prepare new image size to retrieve half-resolution images
image_size = zed.get_resolution()
new_width = image_size.width / 2
new_height = image_size.height / 2

# Declare your sl.Mat matrices
image_zed = sl.Mat(new_width, new_height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
depth_map_zed = sl.Mat()

key = ' '
while key != 113 :
    err = zed.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS :
        # Retrieve the left image, depth image in the half-resolution
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT, sl.MEM.MEM_CPU, int(new_width), int(new_height))
        zed.retrieve_measure(depth_map_zed, sl.MEASURE.MEASURE_DEPTH)
        
        # To recover data from sl.Mat to use it with opencv, use the get_data() method
        # It returns a numpy array that can be used as a matrix with opencv
        testFrame = image_zed.get_data()
        #change colorspaces
        hsvFrame = cv2.cvtColor(testFrame, cv2.COLOR_BGR2HSV)
                
        #set bounds for what is "green" and threshold based on that values
        lowerHSVBound = np.array([60, 50, 30])
        upperHSVBound = np.array([100, 255, 255])
        maskFrame = cv2.inRange(hsvFrame, lowerHSVBound, upperHSVBound)

        #perform morphological transformation to remove noise from the image
        #read more here: https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
        kernel = np.ones((5, 5), np.uint8)
        maskFrame = cv2.morphologyEx(maskFrame, cv2.MORPH_OPEN, kernel)
        
        #get contours of the detected tape
        edgeFrame = cv2.Canny(maskFrame,100,200)
        contourFrame, contours, hierarchy = cv2.findContours(edgeFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                
                #depth of each tape strip and their difference
                right_tape_depth = depth_map_zed.get_value(tape[0], tape[1])[1]
                left_tape_depth =  depth_map_zed.get_value(pos2[0], pos2[1])[1]
                target_depth = (right_tape_depth + left_tape_depth) / 2
                deltad = right_tape_depth - left_tape_depth

                #angle to each of those tapes
                angle_to_right_tape = math.atan((tape[0] - (np.size(testFrame, 1) / 2)) / 350)
                angle_to_left_tape = math.atan((pos2[0] - (np.size(testFrame, 1) / 2)) / 350)
                angle_to_target = math.atan((pixel_x - (np.size(testFrame, 1) / 2)) / 350)
                
                #get real world x offset in meters
                right_tape_x_meters = right_tape_depth * math.tan(angle_to_right_tape)
                left_tape_x_meters = left_tape_depth * math.tan(angle_to_left_tape)
                delta_x_meters = right_tape_x_meters - left_tape_x_meters
                
                correction_angle = math.atan(deltad/delta_x_meters)
                real_x = target_depth * math.tan(angle_to_target)
                cv2.circle(testFrame, (pixel_x, pixel_y), 3, (255, 0, 255), -1)
                cv2.putText(testFrame, "Z: " + str(target_depth) + " X: " + str(real_x) + " A: " + str(math.degrees(correction_angle)), (pixel_x + 15, pixel_y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                    
        #display results
        cv2.imshow("testFrame", testFrame)
        #enabling this can be useful for using a color picker to find exactly what range of HSV
        #works for your image/object
        cv2.imshow("hsvFrame", hsvFrame)
        cv2.imshow("maskframe", maskFrame)
        key = cv2.waitKey(10)

cv2.destroyAllWindows()
zed.close()
