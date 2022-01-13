import numpy as np
import cv2 
import argparse

"""Code for use testing. For verification of calibration factors calculated from homographically perspective corrected images.
Works by analysing image of a chess board whose plane is close to parellel with the camera sensor plane. Distance between camera and chessboard 
is derived from the perspective corrected image. """

parser = argparse.ArgumentParser(description='Verifying pixel calibration using a reference image')
parser.add_argument("--image", type=str, required= False, help='path to image file')
args = parser.parse_args()

image_path = args.image

# Load image
image = cv2.imread(image_path)

# Detect Chessboard

objpoints = []

imgpoints=[]

        # termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Define height and width of chessboard squares
        #self.height = square_height
        #self.width = square_width

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)

objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3), 0)

cv2.imshow('chessboard', gray)
cv2.imshow('blur chessboard', blur)
cv2.waitKey(0)


# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (3,3), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

     # If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
    imgpoints.append(corners)

            # Return an array from the elements of a at given indices
    #points_of_interest = np.ndarray.take(corners2, [[0, 1], [16, 17], [90, 91], [106, 107]])
            #self.points_of_interest = np.ndarray.take(self.corners2, [[0, 1], [2, 3], [16, 17], [18, 19]])
    #print(corners2[47])

    # Draw and display the corners
    cv2.drawChessboardCorners(image, (3,3), corners2, ret)
    cv2.imshow('img', image)
    cv2.waitKey()
    print(corners2)

    x0 = corners2[0][0][0]
    x1 = corners2[8][0][0]
    y0 = corners2[0][0][1]
    y1 = corners2[8][0][1]
    a = corners2[0][0]
    b = corners2[8][0]

    # find width of square in pixels
    w = x1 - x0 

    # height
    h = w
    # Finds euclidean distance between corners defining the edge of a square
    dimension_pix = np.linalg.norm(a - b)





# Define width of square in mm
"""square_dimensions = float(input("Enter height/width of square as measured with ruler (mm): "))


# Find ratio of width in mm to width in pixels
pixel_calibration = square_dimensions/dimension_pix
print(dimension_pix)
print(pixel_calibration)"""

