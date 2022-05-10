
'''
Code for calculating the height, width, and area of an x-ray beam by analysis of images of
exposed phosphor scintillation material

'''
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from region_growing import RegionGrow
import argparse
import math
import logging 
import homography
from skimage import transform
from skimage.io import imread, imshow
from scipy.stats import norm


# load video and select frame averaging method
parser = argparse.ArgumentParser(description='Code for calculating the height, width, and area of an x-ray beam by analysis of images of exposed phosphor scintillation material')
parser.add_argument("--video", type=str, required= True, help='path to image file')

args = parser.parse_args()

start_time = cv2.getTickCount()

# Create log file
logging.basicConfig(filename= 'test.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Load video frames
cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    print("Error opening video")

# Get video information 
fps = cap.get(cv2.CAP_PROP_FPS)

# Convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# To prevent killing
# https://stackoverflow.com/questions/59102833/python-opencv-cv2-videocapture-read-getting-stuck-indefinitely-after-running-t
count = 0
frames = []

# read video
ret,frame = cap.read()

# split video into component frames
while ret:
    ret,frame = cap.read()
    try:
            cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file    
            frames.append(frame)  
            ret,frame = cap.read()
            print('Read a new frame: ', ret)
    except:
            print('Error: missing frame')
            continue
    
    print(count)
    count += 1

# Convert images to 4d ndarray, size(n, nrows, ncols, 3)
frames = np.stack(frames, axis=0)

# Define function to find median or mean image
def find_average(frames):
    return np.mean(frames, axis=0)

def resize_w_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)



# Define function to apply thresholds, is this the best threshold method? 
def apply_threshold(image):

    ret, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return  mask


# Use global threshold to isolate region of beam, output a binary mask 
def mask_img(image):

        # histogram equalisation
        #      #histogram_eq = cv2.equalizeHist(image)
        
        # find G, b*, and hue image 
        # G image
    green_image = image.copy()
    green_image[:,:,0] = 0
    green_image[:,:,2] = 0
    b, green, r = cv2.split(image)

    # write contrast enhanced green
    #cv2.imwrite('contrast_enhanced_green_channel.png', green)
    #
    #histogram_eq_green = cv2.equalizeHist(green)
    #cv2.imshow('Green equalised histogram', green)
    #cv2.waitKey(0)
        

    # find b* image
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L_image, a, b_image = cv2.split(lab_image)       
        
    # Merge G, b*, and hue
    merged = cv2.merge([green, b_image, L_image])
    #merged = cv2.merge([histogram_eq_green, b_image, L_image])
        
    # Reshape the merged image
    #kmeans_image = lab_image.reshape((-1, 3))
    kmeans_image = merged.reshape((-1, 3))

    # Convert to np.float32
    kmeans_image = np.float32(kmeans_image)
       
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    K = 2
    ret, label, center=cv2.kmeans(kmeans_image, K, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
    center = np.uint8(center)
        
    res = center[label.flatten()]
    kmeans_segmented = res.reshape((image.shape))
        #cv2.imshow('kmeans segmented',kmeans_segmented)
        #
        #cv2.waitKey(0)

    # Save unblurred kmeans image
    cv2.imwrite('kmeans_segmented.png', kmeans_segmented)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(kmeans_segmented, (3, 3), 0)
    return blur

def distance_from_camera(focal_length, img_ref_obj_sensor, width_ref_obj):

    distance = (width_ref_obj*focal_length)/img_ref_obj_sensor
    return distance

# Create bounding box function

class BoundingBoxInfo:

    def __init__(self, src, mask, perspective_transform, area_calibration, width_calibration, height_calibration, kernel_size, iterations = 1) -> None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        imgDil = cv2.dilate(mask, kernel, iterations)

        openImg = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, )
        #cv2.imshow("Open Image", openImg)
        #cv2.waitKey(0)
    
        contours, hierarchy = cv2.findContours(openImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cnt = contours[0]

        
   
        # Count number of non-zero pixels in binary mask
        non_zero_pixels = cv2.countNonZero(mask)

        # Create Bounding Box   
        #area = cv2.contourArea(cnt)
   
        area_non_zero_pixels = non_zero_pixels
        rect = cv2.minAreaRect(cnt)
        self.box = cv2.boxPoints(rect)
        self.box = np.int0(self.box)
        (x, y), (width, height), angle = rect

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        print(len(approx))
        x, y, w, h = cv2.boundingRect(approx)

        #self.area = area*area_calibration
    
        self.width = width*width_calibration
        self.height = height*height_calibration
        self.area = self.width*self.height

def distance_from_camera(focal_length, img_ref_obj_sensor, width_ref_obj):

    distance = (width_ref_obj*focal_length)/img_ref_obj_sensor
    return distance

def transform_perspective(frame, homography_transform, image_height=frame_height, image_width=frame_width):
    return cv2.warpPerspective(frame,homography_transform,(image_width,image_height))

# ROI seletor
def select_roi(image):
    # Select rectangular region of interest from average grayscale image that approximately corresponds to the beam area
    # Select ROI
    from_centre = False
    roi = cv2.selectROI(image, from_centre)
    # Crop roi image, needed to create structuring element
    cropped_roi = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    return cropped_roi, roi

def binary_image(image, roi_img, roi):    
    
    image = image.astype('uint8')*255
    # Convert image to greyscale
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load ROI image
    src_mask = roi_img

    # Generate mask from ROI by applying k-means clustering, generates a blurred two-toned colour image
    roi_mask = mask_img(src_mask)

    # Convert the mask to greyscale
    greyscale_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)

    # Apply otsu threshold to roi
    otsu_thresh, otsu_greyscale_roi = cv2.threshold(greyscale_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Save the Otsu (binary) image to a file
    cv2.imwrite('otsu_roi.png', otsu_greyscale_roi)

    # Create black background with dimensions of greyscale image; grey image is now just a black background
    grey_image[:,:] = np.ones(grey_image.shape[:2])

    # Add binary mask to original greyscale image ; grey image is now a black background with white pixels where the beam should be
    grey_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = otsu_greyscale_roi

    # Apply Otsu threshold to masked image to create image to use for seed selection
    ret, binary_mask = cv2.threshold(grey_image, otsu_thresh, 255, cv2.THRESH_BINARY)
        
    return binary_mask
    
# Find median/mean image

frame = cv2.imread('frame231.jpg')
#image = cv2.imread('median.jpg')
#image = cv2.imread('mean.jpg')
image = np.uint8(frame)
cv2.imshow('image', image)
cv2.waitKey(0)
calibration_img = cv2.imread('chessboard.jpg')
calibration_img = np.uint8(calibration_img)
calibration_img_copy = calibration_img.copy()




# Enhance contrast of blue, green, and red channels using histogram equalisation
blue, green, red = cv2.split(image)

# Calibrate dimensions based on perspective corrected image


'Processing to determine area:'
# define homography of image with chessboard pattern
calibrate_homography_img = homography.Homography(image, calibration_image=calibration_img)

# pixel width  (mm/pix)
pixel_width = calibrate_homography_img.pixel_width()

# pixel height
pixel_height = calibrate_homography_img.pixel_height()

# pixel area 
pixel_area = pixel_height*pixel_width

# Width chessboard square in pixels
square_width_pix = calibrate_homography_img.num_pix_chess_square

# Width chessboard square in mm
square_width_mm = calibrate_homography_img.square_dimensions

# Width of sensor pixel (mm)
sensor_pixel_width = 0.0014

# Size of chessboard projection on sensor (mm)
square_width_on_sensor = sensor_pixel_width*square_width_pix

# Distance from ref obj to camera
distance_to_ref_obj = distance_from_camera(3.6, square_width_on_sensor, square_width_mm)


# Correct perspective of image 
homography_transform = calibrate_homography_img.perspective_transform()
corrected_image = transform_perspective(frame, homography_transform)

#cv2.imshow('Calibration Image', calibration_img_copy)
#cv2.waitKey(0)

corrected_chessboard = transform_perspective(calibration_img, homography_transform)
cv2.imwrite('corrected_chessboard.png', corrected_chessboard)

# Apply homography to each frame in frames
corrected_frames = [transform_perspective(frame, homography_transform) for frame in frames[231:261]]

# Show original frame and perspective corrected frame
cv2.imshow('frame', frame)
cv2.imwrite('uncorrected_frame.png', frame)
cv2.imshow('corrected image', corrected_image)
cv2.imwrite('corrected_frame.png', corrected_image)
cv2.waitKey(0)



# select roi
roi_image, roi = select_roi(corrected_image)

# Generate binary mask images from each perspective corrected frame
correct_perspective_binaries = [binary_image(corrected_frame, corrected_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])], roi) for corrected_frame in corrected_frames]

# Bounding boxes lists
bounding_boxes_widths = []
bounding_boxes_heights = []
bounding_boxes_areas = []

# Find the width of each binary by applying bounding box
for correct_perspective_bin in correct_perspective_binaries:
    #cv2.imshow('binary image', correct_perspective_bin)
    #cv2.waitKey(0)
    #bounding_box_img = bounding_box(src= image, perspective_transform=homography_transform, mask= correct_perspective_bin, area_calibration= pixel_area, width_calibration= pixel_width, height_calibration= pixel_height, kernel_size= 3, iterations= 1)
    bounding_boxes = BoundingBoxInfo(src= image, perspective_transform=homography_transform, mask= correct_perspective_bin, area_calibration= pixel_area, width_calibration= pixel_width, height_calibration= pixel_height, kernel_size= 5, iterations= 1)
    # Exclude outliers
    if bounding_boxes.width > 3: 
        bounding_boxes_widths.append(bounding_boxes.width) 
    if bounding_boxes.height > 10: 
        bounding_boxes_heights.append(bounding_boxes.height)
    if bounding_boxes.area > 500: 
        bounding_boxes_areas.append(bounding_boxes.area)

print('here1')
# Average binary
average_binary = find_average(frames= correct_perspective_binaries)
average_binary = np.uint8(average_binary)
cv2.imshow('average', average_binary)
cv2.waitKey(0)
print('here2')
# Find mean width
mean_width = np.mean(bounding_boxes_widths)

# Standard deviation
std_w = np.std(bounding_boxes_widths)

# Find mean height
mean_height = np.mean(bounding_boxes_heights)

# Standard deviation
std_h = np.std(bounding_boxes_heights)

# Find mean area
mean_area = np.mean(bounding_boxes_areas)

# Standard deviation
std_a = np.std(bounding_boxes_areas)

# Function to apply bounding box to correct_perspective_binaries
def bounding_box(src, mask, kernel_size, perspective_transform, area_calibration, width_calibration, height_calibration, average_width= mean_width, average_height= mean_height, average_area= mean_area, 
                    std_a= std_a, std_h =std_h, std_w= std_w,  iterations = 1, image_height=frame_height, image_width=frame_width):

    # Apply inverse homography to mask, so it is in the uncorrected perspective
    mask = cv2.warpPerspective(mask, perspective_transform,(image_width,image_height))

    # Square shaped Structuring Element
    #kernel = np.ones((kernel_size, kernel_size))
    # Cross shaped structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (kernel_size, kernel_size))
    #imgDil = cv2.dilate(mask, kernel, iterations)
    openedImg = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("Opened Image", openedImg)
    #cv2.waitKey(0)


    contours, hierarchy = cv2.findContours(openedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(closedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    # Correct video frame perspective 
    #corrected_src = cv2.warpPerspective(src, perspective_transform,(image_width,image_height))
   
    # Count number of non-zero pixels in binary mask
    non_zero_pixels = cv2.countNonZero(mask)

    # Create Bounding Box   
    area = cv2.contourArea(cnt)
   
    area_non_zero_pixels = non_zero_pixels
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    (x, y), (height, width), angle = rect

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    print(len(approx))
    x, y, w, h = cv2.boundingRect(approx)

    # Can I correct perspective of binary mask rather than src? Such that the output video shows the original perspective but with accurate measurements.
    
    # Draw the contours and display info
    cv2.drawContours(src,[box],0,(0,255, 0),2)
    cv2.putText(src, "Bounding Box Area: {0:.3g}".format(average_area) + "+/-{0:.3g} mm^2".format(std_a), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(src, "Bounding Box Width: {0:.3g}".format(average_width) + "+/- {0:.3g} mm".format(std_w), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(src, "Bounding Box Height: {0:.3g}".format(average_height) + "+/- {0:.3g} mm".format(std_h), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    return src

print(type(image))
print('here3!')
# Apply contours to cropped_histogram_equalised_product_image to generate bounding box and display area

# Inverse homography
homography_inverse = np.linalg.inv(homography_transform)
masked_frame = bounding_box(src= image, perspective_transform=homography_inverse, mask= average_binary, area_calibration= pixel_area, width_calibration= pixel_width, 
                                height_calibration= pixel_height, kernel_size= 5, iterations= 1)
cv2.imshow('Masked Frame', masked_frame)
cv2.waitKey(0)
cv2.imwrite('masked_frame_w_bb.png', masked_frame)





############################################################################

# Time to create bounding box/masked image

end_time =cv2.getTickCount()
'''bounding_box_time = cv2.getTickCount()
bounding_box_time = (bounding_box_time - time_after_seed_selection)/cv2.getTickFrequency()
logging.info('Time of masked image creation: {} s'.format(bounding_box_time))'''

##################################################################################


total_time = (end_time - start_time)/cv2.getTickFrequency()
logging.info('Time of masked image creation from start: {} s'.format(total_time))



# Function to write video

def write_masked_video(frames_list, mask, area_calibration, width_calibration, height_calibration):
    

    # Load Video Stream
    video_cap = cv2.VideoCapture(args.video)

    # Write to video when called
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    # Frame counter

    count = 0
    # Process each frame separately

    for frame in frames_list:
        try:

        # Read frame
            ret, masked_frame = video_cap.read()
            count += 1

        # Apply bounding box to frame 
            masked_frame = bounding_box(src=masked_frame, perspective_transform= homography_transform, mask=mask, area_calibration= area_calibration, width_calibration= width_calibration, height_calibration= height_calibration, kernel_size= 3, iterations=1)
            #cv2.imshow('masked frame', masked_frame)
            #cv2.waitKey(0)
        # Apply contours
            #masked_frame = fitcontours(masked_frame, mask)

        # Write frame with mask to video
            out.write(masked_frame)
    
        except ValueError:
            print('Last frame read: frame%d', count)
            print('failed')
            print('Video frame width and height: ', (frame_width, frame_height))
            print('Mask dimensions: ', mask.shape[:2])
            print('Masked frame dimensions: ', masked_frame.shape[:2])
            break

    cap.release()
    cv2.destroyAllWindows()

# Create video with masked frames
write_masked_video(frames_list= frames, mask= average_binary, area_calibration= pixel_area, width_calibration= pixel_width, height_calibration= pixel_height)


logging.info('Area Calibration Factor: {}'.format(pixel_area))
logging.info('Width Calibration Factor: {}'.format(pixel_width))
logging.info('Square width: {}'.format(square_width_pix))
logging.info('Square width (mm): {}'.format(square_width_mm))
logging.info('Height Calibration Factor: {}'.format(pixel_height))
logging.info('Frame Width: {}'.format(frame_width))
logging.info('Frame Height: {}'.format(frame_height))
logging.info('Distance to transformed plane: {}'.format(distance_to_ref_obj))

# Importing library
import csv
  
n_bins = 30

# Generate two normal distributions
dist1 = bounding_boxes_widths
dist2 = bounding_boxes_heights
dist3 = bounding_boxes_areas



fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)


axs[0].hist(dist1, bins=n_bins)
axs[0].set_xlabel('Width (mm)')
axs[0].set_title('n= {}'.format(len(bounding_boxes_widths)))

axs[1].hist(dist2, bins=n_bins)
axs[1].set_xlabel('Height (mm)')
axs[1].set_title('n= {}'.format(len(bounding_boxes_heights)))

axs[2].hist(dist3, bins=n_bins)
axs[2].set_xlabel('Area (mm^2)')
axs[2].set_title('n= {}'.format(len(bounding_boxes_areas)))
#axs.set_title('Dimension distribution')
plt.show()

#fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)




print(bounding_boxes_widths)
print(len(bounding_boxes_widths))
print(bounding_boxes_heights)
print(len(bounding_boxes_heights))

