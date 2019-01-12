import sys
import pyzed.sl as sl
import numpy as np
import cv2

zed = sl.Camera()

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE
init.coordinate_units = sl.UNIT.UNIT_METER


# Open the camera
err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS :
	print(repr(err))
	zed.close()
	exit(1)

runtime = sl.RuntimeParameters()
runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

# Prepare new image size to retrieve half-resolution images
image_size = zed.get_resolution()
new_width = image_size.width /2
new_height = image_size.height /2

# Declare your sl.Mat matrices
image_zed = sl.Mat(new_width, new_height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
depth_image_zed = sl.Mat(new_width, new_height, sl.MAT_TYPE.MAT_TYPE_8U_C4)

key = ' '
while key != 113 :
	err = zed.grab(runtime)
	if err == sl.ERROR_CODE.SUCCESS :
		# Retrieve the left image, depth image in the half-resolution
		zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT, sl.MEM.MEM_CPU, int(new_width), int(new_height))
		zed.retrieve_image(depth_image_zed, sl.VIEW.VIEW_DEPTH, sl.MEM.MEM_CPU, int(new_width), int(new_height))

		# To recover data from sl.Mat to use it with opencv, use the get_data() method
		# It returns a numpy array that can be used as a matrix with opencv
		image_ocv = image_zed.get_data()
		depth_image_ocv = depth_image_zed.get_data()

		cv2.imshow("Image", image_ocv)
		cv2.imshow("Depth", depth_image_ocv)

		key = cv2.waitKey(10)

cv2.destroyAllWindows()
zed.close()
