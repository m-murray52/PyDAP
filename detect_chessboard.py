import numpy as np
import cv2 
import glob
import argparse


parser = argparse.ArgumentParser(description='Module to determined correct chessboard coordinates for homography')
parser.add_argument('--image', type=str, required= False, help='Input image')

args = parser.parse_args()


"""# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)

objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


#images = glob.glob('*.jpg')
#for fname in images:
img = cv2.imread('/home/michael/Pictures/Screenshot_20211219_013904.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    # If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray,corners, (7,7), (-1,-1), criteria)
    imgpoints.append(corners)
        # Draw and display the corners
    cv2.drawChessboardCorners(img, (8,6), corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()
#print(corners)
print(corners2)
print(len(corners2))
print(corners2.shape)
print(type(corners2))

# Return an array from the elements of a at given indices
new_array = np.ndarray.take(corners2, [[[0, 1]], [[2, 3]], [[16, 17]], [[18, 19]]])
#print(corners2[0])
print(new_array)

# create projection (h, w) = (85, 85)
x0 = new_array[0][0][0] 
y0 = new_array[0][0][1]


print(new_array[1])

# Target projection 
projection = [[x0, y0], [x0, y0 +85], [x0 - 85, y0], [x0 -85, y0 +85]]

print(projection)"""

class Chessboard:
    """
    Detects chessboard pattern corners and outputs corrected chessboard 
    corner coordinates
    """
    

    def __init__(self, img, objpoints = [], imgpoints=[]) -> None:
       
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((6*7,3), np.float32)

        self.objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
        
        self.img = cv2.imread(img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, self.corners = cv2.findChessboardCorners(self.gray, (8,6), None)

     # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(self.objp)
            self.corners2 = cv2.cornerSubPix(self.gray, self.corners, (7,7), (-1,-1), self.criteria)
            imgpoints.append(self.corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(self.img, (8,6), self.corners2, ret)
        cv2.imshow('img', self.img)
        cv2.waitKey()

    def projection(self, height=85, width=85) -> np.ndarray:
        # Return an array from the elements of a at given indices
        new_array = np.ndarray.take(self.corners2, [[[0, 1]], [[2, 3]], [[16, 17]], [[18, 19]]])
    
        # create projection, e.g. (h, w) = (85, 85)
        # Co-ordinates of top right corner (will change later to top left)
        x0 = new_array[0][0][0] 
        y0 = new_array[0][0][1]


        print(new_array[1])

        # Target projection 
        projection = [[x0, y0], [x0, y0 + height], [x0 - width, y0], [x0 -width, y0 + height]]
        return projection

chess_img = Chessboard(args.image)
print(chess_img.projection())