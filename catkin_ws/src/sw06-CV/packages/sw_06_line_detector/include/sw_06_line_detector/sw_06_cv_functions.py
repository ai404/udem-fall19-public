import cv2
import numpy as np
from scipy.ndimage import binary_dilation

import rospy

## Software Exercise 6: Choose your category (1 or 2) and replace the cv2 code by your own!

## CATEGORY 1
def inRange(hsv_image, low_range, high_range):
	# reshape thershold vectors
	low_range = low_range.reshape(1,1,-1)
	high_range = high_range.reshape(1,1,-1)

	# >= because of inclusive boundaries
	return np.prod((hsv_image>=low_range)&(hsv_image<=high_range),-1)*255
	#return cv2.inRange(hsv_image, low_range, high_range)

def bitwise_or(bitwise1, bitwise2):
	return bitwise1 | bitwise2
	#return cv2.bitwise_or(bitwise1, bitwise2)

def bitwise_and(bitwise1, bitwise2):
	return bitwise1 & bitwise2
	#return cv2.bitwise_and(bitwise1, bitwise2)

def getStructuringElement(shape, size):

    MORPH_RECT = 0
    MORPH_CROSS = 1
    MORPH_ELLIPSE = 2
    # calculate center coordinates
    x, y = size
    center_x = x//2
    center_y = y//2
    inv_r2 = 1./center_y**2 if center_y!=0 else 0

    # init output array
    np_obj = np.zeros(size,dtype=np.uint8)

    if shape == MORPH_RECT:
        # set all values to 1
        np_obj+= 1
    elif shape == MORPH_CROSS:
        # from a plus form with ones
        np_obj[center_x,:] = 1
        np_obj[:,center_y] = 1
    elif shape == MORPH_ELLIPSE:
        # form an ellipse of ones
        for i in range(y):
            j1, j2 = 0, 0
            dy = i - center_y
            if np.abs(dy)<=center_y:
                dx = int(np.round(center_x*np.sqrt((center_y**2 - dy**2)*inv_r2)))
                j1 = max([center_x - dx, 0])
                j2 = min([center_x + dx +1, x])

            np_obj[j1:j2,i] = 1

    return np_obj.T
	#return cv2.getStructuringElement(shape, size)

def dilate(bitwise, kernel):
    w,h = kernel.shape
    pad_w = w//2
    pad_h = h//2
	# pad the bitwise matrix with zeros
    pad_bitwise  = np.pad(bitwise, [(pad_w, pad_w), (pad_h, pad_h)], mode='constant', constant_values=0)
    def process(i, j):
		"""
			git the matrix with the same size as the kernel, centered at i,j coordinates in bitwise
		"""
        shrunk = pad_bitwise[ i:i + w , j:j + h]
        return np.any(kernel & shrunk)
    return np.array([[ process(i,j) for j in range(bitwise.shape[1])] for i in range(bitwise.shape[0])],dtype=np.uint8)
	#return binary_dilation(bitwise, kernel)
	#return cv2.dilate(bitwise, kernel)



## CATEGORY 2
def Canny(image, threshold1, threshold2, apertureSize=3):
	return cv2.Canny(image, threshold1, threshold2, apertureSize=3)


## CATEGORY 3 (This is a bonus!)
def HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap):
	return cv2.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap)