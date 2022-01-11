import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.core.fromnumeric import shape
import detect_chessboard
from skimage import transform
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import logging 

"""logging.basicConfig(filename= 'test_homography.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
# Load image
img1 = cv2.imread('chessboard.jpg')

# detect chessboard pattern
chess_pattern = detect_chessboard.Chessboard(img1, 10.0, 10.0)

# extract points of interest, i.e., the corners of the chessboard pattern
points_of_interest = chess_pattern.points_of_interest

# define target coordinates for the corrected chessboard image. Essentially "telling"
# the computer what the correct homography is
projection = chess_pattern.projection()
print(points_of_interest)
print(projection)
print(type(points_of_interest))
print(type(projection))
print(shape(points_of_interest))
print(shape(projection))

logging.info('Points of Interest: {}'.format(points_of_interest))
logging.info('Projection: {}'.format(projection))


# estimate the homographic transform needed to correct the perspective of the image
tform = transform.estimate_transform('projective', points_of_interest, projection)

# perform perspective correction 
tf_img_warp = transform.warp(img1, tform.inverse, mode = 'symmetric')

# Display image
cv2.imshow('warped img', tf_img_warp)
cv2.imshow('unwarped img', img1)
cv2.waitKey()"""

class Homography:
    """Performs homographic perspective correction on images such that the reference object (chessboard) and image planes are 
    approximately parallel"""

    def __init__(self, frame, calibration_image) -> None:
        self.frame = frame
        #self.width_calibration = width_calibration
        #self.height_calibration = height_calibration

        self.square_dimensions = float(input("Enter height/width of square as measured with ruler (mm): "))
        
        # dimensions in units of pixel
        
        self.chess_pattern = detect_chessboard.Chessboard(calibration_image)
        self.points_of_interest = np.float32(self.chess_pattern.points_of_interest)
        self.projection = np.float32(self.chess_pattern.projection())

        # dimensions chessboard square
        self.num_pix_chess_square = self.chess_pattern.dimension_pix

    def perspective_transform(self):
        # estimate the homographic transform needed to correct the perspective of the image
        #tform = transform.estimate_transform('projective', self.points_of_interest, self.projection)

        M = cv2.getPerspectiveTransform(self.points_of_interest, self.projection)
        # perform perspective correction on whatever the current image set is 
        #tf_img_warp = transform.warp(self.frame, tform.inverse, mode = 'symmetric')

        #cv2.imshow('Transformed image', tf_img_warp)
        #cv2.waitKey(0)
        return M
        

    def pixel_width(self):
        return self.square_dimensions/self.chess_pattern.dimension_pix

    def pixel_height(self):
        return self.square_dimensions/self.chess_pattern.dimension_pix

    
    



#print(chess_pattern.corners2)
#print(shape(chess_pattern.corners2))


"""
# Alternative, opencv docs method (https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html#tutorial_homography_Demo1): 

ret1, corners1 = cv2.findChessboardCorners(img1, patternSize)
ret2, corners2 = cv2.findChessboardCorners(img2, patternSize)

H, _ = cv2.findHomography(corners1, corners2)
print(H)
    img1_warp = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))"""