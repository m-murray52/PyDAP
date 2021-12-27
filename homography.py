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


img1 = cv2.imread('/home/michael/Pictures/Screenshot_20211219_013904.png')
chess_pattern = detect_chessboard.Chessboard('/home/michael/Pictures/Screenshot_20211219_013904.png')
points_of_interest = chess_pattern.points_of_interest
projection = chess_pattern.projection()
print(points_of_interest)
print(projection)
print(type(points_of_interest))
print(type(projection))
print(shape(points_of_interest))
print(shape(projection))

tform = transform.estimate_transform('projective', points_of_interest, projection)
tf_img_warp = transform.warp(img1, tform.inverse, mode = 'symmetric')
cv2.imshow('img', tf_img_warp)
cv2.waitKey()

"""color = 'green'
patches = []
fig, ax = plt.subplots(1,2, figsize=(15, 10), dpi = 80)
for coordinates in (points_of_interest + projection):
    patch = Circle((coordinates[0],coordinates[1]), 10, 
                    facecolor = color)
    patches.append(patch)
for p in patches[:4]:
    ax[0].add_patch(p)
ax[0].imshow(img1)

for p in patches[4:]:
    ax[1].add_patch(p)
ax[1].imshow(np.ones((img1.shape[0], img1.shape[1])))"""

"""ret1, corners1 = cv2.findChessboardCorners(img1, patternSize)
ret2, corners2 = cv2.findChessboardCorners(img2, patternSize)

H, _ = cv2.findHomography(corners1, corners2)
print(H)
    img1_warp = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))"""