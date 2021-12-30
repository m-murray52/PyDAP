import numpy as np
import cv2 
import glob
#import argparse


#parser = argparse.ArgumentParser(description='Module to determined correct chessboard coordinates for homography')
#parser.add_argument('--image', type=str, required= False, help='Input image')

#args = parser.parse_args()


class Chessboard:
    """
    Detects chessboard pattern corners and outputs corrected chessboard 
    corner coordinates
    """
    

    def __init__(self, img, square_height, square_width, objpoints = [], imgpoints=[]) -> None:
       
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Define height and width of chessboard squares
        self.height = square_height
        self.width = square_width

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((6*7,3), np.float32)

        self.objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
        
        self.img = img
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('chessboard', self.gray)
        cv2.waitKey(0)
        # Find the chess board corners
        ret, self.corners = cv2.findChessboardCorners(self.gray, (8,6), None)

     # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(self.objp)
            self.corners2 = cv2.cornerSubPix(self.gray, self.corners, (7,7), (-1,-1), self.criteria)
            imgpoints.append(self.corners)

            # Return an array from the elements of a at given indices
            self.points_of_interest = np.ndarray.take(self.corners2, [[0, 1], [14, 15], [80, 81], [94, 95]])
            #self.points_of_interest = np.ndarray.take(self.corners2, [[0, 1], [2, 3], [16, 17], [18, 19]])
            print(self.corners2[47])

        # Draw and display the corners
        cv2.drawChessboardCorners(self.img, (8,6), self.corners2, ret)
        cv2.imshow('img', self.img)
        cv2.waitKey()
        print(self.corners2)

    def projection(self) -> np.ndarray:
        
        height = self.height
        width = self.width

        # create projection, e.g. (h, w) = (85, 85)
        # Co-ordinates of top right corner (will change later to top left)
        x0 = self.points_of_interest[0][0] 
        y0 = self.points_of_interest[0][1]


        #print(self.points_of_interest)

        # Target projection 
        # height is multiplied by 7 because 7 squares are being used, since a n 8x6 chessboard is being detected.
        # Likewise for width with 5 squares. 
        projection = np.array([[x0, y0], [x0, y0 + 7*height], [x0 - 5*width, y0], [x0 - 5*width, y0 + 7*height]])
        #projection = np.array([[x0, y0], [x0, y0 + height], [x0 - width, y0], [x0 - width, y0 + height]])
        return projection

#chess_img = Chessboard(args.image)
#print(chess_img.projection())
#print(type(chess_img.projection()))