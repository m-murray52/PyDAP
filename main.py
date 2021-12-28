#!source env/bin/activate

'''
Code for calculating the height, width, and area of an x-ray beam by analysis of images of
exposed phosphor scintillation material


Usage:
  main.py [<video>] [<average_image_method>]

  Click on the image to set seed point

Keys:
  f     - toggle floating range
  c     - toggle 4/8 connectivity
  ESC   - exit
'''
import cv2 
import numpy as np 

#from region_growing import RegionGrow
import argparse
import math
import logging 
import homography


# load video and select frame averaging method
parser = argparse.ArgumentParser(description='Code for calculating the height, width, and area of an x-ray beam by analysis of images of exposed phosphor scintillation material')
parser.add_argument('--video', type=str, required= False, help='path to image file')
parser.add_argument('--average', type=str, required= False, help='select method for averaging frames: type mean" or "median" or "no" for no average calculation, instead an image near middle of recording is selected')
parser.add_argument("--threshold", type=str, required=False, help="To estimate 25max intensity type '25max'. By default a user selected threshold is used via trackbar.")    
parser.add_argument("--method", type=str, required= True, help="select colour filtering (colour), greyscale (grey), or 'kmeans' to use kmeans clustering")
parser.add_argument("--gradient", type=str, required= False, help="type 'yes' to use edge enhanced threshold, for use with grey method")

parser.add_argument("--output", type=str, required=False, help='enter the name of the output video including format (.mp4 or .avi)')
parser.add_argument("--distance", type=float, required=True, help='Enter the distance from the camera to the phosphor sheet in mm')
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
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# To prevent killing
# https://stackoverflow.com/questions/59102833/python-opencv-cv2-videocapture-read-getting-stuck-indefinitely-after-running-t
count = 0
frames = []


        

# read video
ret,frame = cap.read()

#Resize Frames to fit screen
#frame = ResizeWithAspectRatio(frame, height=1280)




# split video into component frames

while ret:
    ret,frame = cap.read()
    #if count >= 200:
    try:
        cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file    
        frames.append(frame)  
        ret,frame = cap.read()
        print('Read a new frame: ', ret)
    except:
        print('Error: missing frame')
        continue
    count += 1



# calculate median image from frames
# Convert images to 4d ndarray, size(n, nrows, ncols, 3)
frames = np.stack(frames, axis=0)


# Ask user to provide approximate distance from phosphor to camera in mm
distance = args.distance

# Convert distance from string to float
distance = float(distance)


# Define function to find median or mean image

def find_average(frames, average_type):
    if average_type == 'median':
        return np.median(frames[290:350], axis=0)

    elif average_type == 'mean':
        return np.mean(frames[290:350], axis=0)

    elif average_type == 'no':
        # Middle frame index, beam is likely to be on
        frame_index = len(frames)//2
        return frames[frame_index]

    elif average_type == 'subtraction':
        # Subtract frame from beginnig from frame near middle
        # Middle frame index, beam is likely to be on
        frame_index = len(frames)//2
        middle_frame = frames[frame_index]
        sub = middle_frame.copy()
        cv2.subtract(middle_frame, frames[0], sub)
        cv2.imshow('Subtraction', sub)
        cv2.waitKey(0)
        return sub 

    elif average_type == 'difference':
        frame_index = len(frames)//2
        middle_frame = frames[frame_index]
        diff = middle_frame.copy()
        cv2.absdiff(middle_frame, frames[0], diff)
        cv2.imshow('Difference', diff)
        cv2.waitKey(0)
        return diff

# ROI seletor
def select_roi(image):
    # Select rectangular region of interest from average grayscale image that approximately corresponds to the beam area
    # Select ROI
    from_centre = False
    roi = cv2.selectROI(image, from_centre)
    # Crop roi image, needed to create structuring element
    cropped_roi = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    return cropped_roi, roi


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




# Define function to apply white top hat transform 
def apply_top_hat(image, structure_element_image):
    try: 

    # Use roi to create structuring element for use in white top-hat transform
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, structure_element_image.shape)

    except ValueError:
        print('Problem with kernel. Kernel height and width:' )
    # Apply white top hat tranform to input grayscale image to reduce background illumination
    white_top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, structuring_element)
    return white_top_hat




# Define function to apply thresholds, is this the best threshold method? 
def apply_threshold(image):

    ret, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return  mask




# Use global threshold to isolate region of beam, output a binary mask 
def binary_mask_img(method, gradient, image):

    # Callback functions
    def thresh(*args):
        # Get the threshold from the trackbar

        ret, binary_img = cv2.threshold(resize_converted_colour_image, args[0], 255, cv2.THRESH_BINARY)
        cv2.imshow(windowName, binary_img)
         
    

    # Function to find 25% of maximum intensity  pixel for use as threshold
    def quarter_max(grey_image):
        #smallest = np.amin(grey_image)
        # Apply histogram equalisation
        equ = cv2.equalizeHist(grey_image)
        biggest = np.amax(equ)
        thresh25 = 0.25*biggest
        return thresh25



    if method == 'colour':


        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L_image, a, b_image = cv2.split(image)       
        converted_colour_image = b_image



        # Need to resize image to view on screen
        resize_converted_colour_image = resize_w_aspect_ratio(converted_colour_image, height= 300)


        # Select Threshold method, applied to greyscale b* image

        # Create Track bar
        maxThresh = 255
        th = 100
        windowName = "Resized Image"
        trackbarValue = "Threshold"






        # Create a window to display results and  set the flag to Autosize
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

        # Create trackbar and associate a callback function
        cv2.createTrackbar(trackbarValue, windowName, th, maxThresh, thresh)

        # Log the processing time 
        tick_count1 = cv2.getTickCount()
        time_until_trackbar = (tick_count1 - time_after_select_roi)/cv2.getTickFrequency()
        logging.info('Time until trackbar window displayed: {} s'.format(time_until_trackbar))

        cv2.imshow(windowName, resize_converted_colour_image)



        cv2.waitKey(0)

        global time_threshold_selection
        time_threshold_selection = cv2.getTickCount()

        # After closing image store trackbar value
        trackbar_pos = cv2.getTrackbarPos(trackbarValue, windowName)
        logging.info('Threshold value: {}'.format(trackbar_pos))

        ret, mask = cv2.threshold(converted_colour_image, trackbar_pos, 255, cv2.THRESH_BINARY)

        cv2.destroyAllWindows()

        # Show Mask
        
        #cv2.imshow('Mask', mask)
        #cv2.imwrite('binary_mask.png', mask)
        #cv2.imwrite('converted_colour_image.png', converted_colour_image)
        #cv2.waitKey(0)

        return mask

    elif method == 'grey':
       
      
        ret, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return mask    
       
    
    elif method == 'kmeans':
        
        # histogram equalisation
        #histogram_eq = cv2.equalizeHist(image)
        
        # find G, b*, and hue image 
        # G image
        green_image = image.copy()
        green_image[:,:,0] = 0
        green_image[:,:,2] = 0
        b, green, r = cv2.split(image)

        # write contrast enhanced green
        #cv2.imwrite('contrast_enhanced_green_channel.png', green)
        #
        histogram_eq_green = cv2.equalizeHist(green)
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
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, label, center=cv2.kmeans(kmeans_image, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        
        res = center[label.flatten()]
        kmeans_segmented = res.reshape((image.shape))
        cv2.imshow('kmeans segmented',kmeans_segmented)
        #
        cv2.waitKey(0)

        # Save unblurred kmeans image
        cv2.imwrite('kmeans_segmented_no_blur.png', kmeans_segmented)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(kmeans_segmented, (3, 3), 0)
        return blur




# Create bounding box function
def bounding_box(src, mask, area_calibration, width_calibration, height_calibration, kernel_size, iterations = 1):

    # Square shaped Structuring Element
    #kernel = np.ones((kernel_size, kernel_size))
    # Cross shaped structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS , (kernel_size, kernel_size))
    imgDil = cv2.dilate(mask, kernel, iterations)
    #closedImg = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Closed Image", closedImg)
    #cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(closedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

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

    calibrated_area = area*area_calibration
    # had to swap height and width calibration because video is rotated
    calibrated_width = width*height_calibration
    calibrated_height = height*width_calibration
    calibrate_area_non_zero = area_non_zero_pixels*area_calibration

    #cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #can also use cv2.connectedComponentsWithStats
    #cv2.drawContours(src[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])],[box],0,(0,255, 0),2)

    #logging.info('Time until seed selection window since threshold selection: {} s'.format(time_until_region_growing))

    cv2.drawContours(src,[box],0,(0,255, 0),2)
    cv2.putText(src, "Bounding Box Area: {0:.3g}".format(calibrated_area) + " mm^2", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src, "Bounding Box Area (px): " + str(int(area)) + " px", (20,  80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src, "Bounding Box Height: {0:.3g}".format(calibrated_width) + " mm", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src, "Bounding Width: {0:.3g}".format(calibrated_height) + " mm", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src, "Beam Area: {0:.3g}".format(calibrate_area_non_zero) + " mm^2", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src, "Number non-zero pixels: " + str(int(area_non_zero_pixels)) + " px", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src, "Area Calibration Factor: {0:.3g}".format(area_calibration) + " mm^2/px", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src, "Width Calibration Factor: {0:.3g}".format(height_calibration) + " mm/px", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src, "Height Calibration Factor: {0:.3g}".format(width_calibration) + " mm/px", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    
    return src




def calibrate_height(distance, frame_height):
    # Function to calculate area per pixel based on camera FOV, focal length, and distance of object from lens. 
    # Area of camera sensor is 3.76 x 2.74 mm according to docs
    y = 2.74
    # proportion of sensor height used
    f_h = frame_height/1944
    

    # Find the magnification factor(M) based on distance from phosphor to lens and lens focal length
    # Focal length of Raspberry Pi Camera V1.0 is 3.60 mm +/- 0.01
    focal_length = 3.60

    # Magnification is the ratio of the distance to focal length
    M = distance/focal_length

    # FOV height (mm) 
    h = M*y*f_h

    



    # Calibration factor vertical (y) (mm/pixel)
    Cy = h/frame_height

    return Cy




def calibrate_width(distance, frame_width):
    # Function to calculate area per pixel based on camera FOV, focal length, and distance of object from lens. 
    # Area of camera sensor is 3.76 x 2.74 mm according to docs
    x= 3.76
    

    # Find the magnification factor(M) based on distance from phosphor to lens and lens focal length
    # Focal length of Raspberry Pi Camera V1.0 is 3.60 mm +/- 0.01
    focal_length = 3.60

    # proportion of sensor width used
    f_w = frame_width/2592
    #f_w = 1296/2592

    # Magnification is the ratio of the distance to focal length
    M = distance/focal_length


    # The width (mm) of the FOV is
    w = M*x*f_w


    # Calibration factor horizontal (x) (mm/pixel)
    Cx = w/frame_width 

    return Cx




def calibrate_area(pixel_width, pixel_height):
    # Find mm^2/pixel

    C_area = pixel_width*pixel_height 

    # Calibrate contour area
    return C_area




# Find median/mean image

if args.average == 'median' or args.average == 'mean':
    # Take average between frame 290 and 350 since the beam is believed to be clearly visible in this range
    image = find_average(frames= frames, average_type= args.average)
    image = np.uint8(image)

else:
    frame = cv2.imread('frame360.jpg')
#image = cv2.imread('median.jpg')
#image = cv2.imread('mean.jpg')
    image = np.uint8(frame)





# Enhance contrast of blue, green, and red channels using histogram equalisation
blue, green, red = cv2.split(image)




# histogram equalisation
histogram_eq_blue = cv2.equalizeHist(blue)
histogram_eq_green = cv2.equalizeHist(green)
histogram_eq_red = cv2.equalizeHist(red)




# Merge to form contrast enhanced image
'''contrast_enhanced = cv2.merge([histogram_eq_blue, histogram_eq_green, histogram_eq_red])
cv2.imshow('Contrast enhanced', contrast_enhanced)
cv2.waitKey(0)
cv2.imwrite('contrast_enhanced_colour.png', contrast_enhanced)'''



# Convert the median/mean image to grayscale
#grey_image = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2GRAY)
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imwrite('greyscale_beam.png', grey_image)




# Time until ROI selection
time_roi_selection_window = cv2.getTickCount()
time_roi_selection_window = (time_roi_selection_window - start_time)/cv2.getTickFrequency()
logging.info('Time ROI selection window: {} s'.format(time_roi_selection_window))

# Select ROI
#roi_image, roi = select_roi(contrast_enhanced)
roi_image, roi = select_roi(image)




# Start tick counter
time_after_select_roi = cv2.getTickCount()







# if statement to select colour image or greyscale processing
if args.method == 'grey':

    # Grey ROI image
    grey_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('grey_roi.png', grey_roi)
    # Apply white top hat to grey ROI and pass to binary mask function
    # Apply white top hat to grey image to reduce background illumination
    white_top_hat_image = apply_top_hat(grey_image, grey_roi)
    cv2.imwrite('white_top_hat_transform.png', white_top_hat_image)

  

    # Crop white top hat image  
    cropped_white_top_hat = white_top_hat_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]



    # Source mask image
    src_mask = cropped_white_top_hat
    cv2.imwrite('white_top_hat.png', white_top_hat_image)
    cv2.imwrite('cropped_white_top_hat.png', cropped_white_top_hat)

elif args.method == 'colour':

    # Grey ROI image
    grey_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('grey_roi.png', grey_roi)
    src_mask = roi_image

elif args.method == 'kmeans':
    src_mask = roi_image





# Generate binary mask image from L*a*b* space image or region growing or combination of the two
#mask_image = maskImg(args.method, roi_image)

mask_image = binary_mask_img(args.method, args.gradient, src_mask)

cv2.imwrite('unprocessed_binary_mask.png', mask_image)


##########################################################################################
# Start second timer for colour processing, if using greyscale don't start new timer






###########################################################################################


# convert mask image to greyscale if method is kmeans
if args.method == 'kmeans':
    greyscale_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('grayscale kmeans', greyscale_mask)
    cv2.imwrite('greyscale_mask.png', greyscale_mask)



else:

    # Find product of the region of interest of the greyscale image and the binary mask image (b* or grey)
    greyscale_mask = cv2.bitwise_and(grey_roi, mask_image)
    cv2.imwrite('greyscale_mask.png', greyscale_mask)



#cv2.imshow('Cropped product of Grey ROI and Mask', greyscale_mask)
#cv2.waitKey(0)



# Estimate IEC defined beam area
if args.threshold == '25max':

   


    #Threshold using 25% of maximum greyscale mask
    biggest = np.amax(greyscale_mask)
    threshold = 0.25*biggest
    ret,thresh_25_max = cv2.threshold(greyscale_mask, threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('>25% Max Intensity', thresh_25_max)
    #cv2.waitKey(0)
    cv2.imwrite('25_max.png', thresh_25_max)
    #print(threshold, biggest)





    # Create black background with dimensions of greyscale image
    grey_image[:,:] = np.ones(grey_image.shape[:2])

    grey_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = thresh_25_max
    # Add greyscale_25_mask to original greyscale image 
    #grey_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = greyscale_25_mask

    # Apply Otsu threshold to masked image to create image to use for seed selection
    img_for_seed = grey_image

elif args.method == 'kmeans':

    # Apply otsu to roi, then add to greyscale image with black background
    otsu_thresh, otsu_greyscale_roi = cv2.threshold(greyscale_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite('otsu_roi.png', otsu_greyscale_roi)
# image.shape[:2]
    # Create black background with dimensions of greyscale image
    grey_image[:,:] = np.ones(grey_image.shape[:2])

    # Add greyscale_mask to original greyscale image 
    #grey_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = greyscale_mask
    grey_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = otsu_greyscale_roi

    # Apply Otsu threshold to masked image to create image to use for seed selection
    ret, img_for_seed = cv2.threshold(grey_image, otsu_thresh, 255, cv2.THRESH_BINARY)
    #img_for_seed  = grey_image
    cv2.imwrite('image_for_seed_growth.png', img_for_seed)
    cv2.imshow('Mask image', img_for_seed)
    cv2.waitKey(0)

else:
    # Apply otsu to roi, then add to greyscale image with black background
    otsu_thresh, otsu_greyscale_roi = cv2.threshold(greyscale_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite('otsu_roi.png', otsu_greyscale_roi)
# image.shape[:2]
    # Create black background with dimensions of greyscale image
    grey_image[:,:] = np.ones(grey_image.shape[:2])

    # Add greyscale_mask to original greyscale image 
    #grey_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = greyscale_mask
    grey_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = otsu_greyscale_roi

    # Apply Otsu threshold to masked image to create image to use for seed selection
    #ret, img_for_seed = cv2.threshold(grey_image, otsu_thresh, 255, cv2.THRESH_BINARY)
    img_for_seed  = grey_image
    cv2.imwrite('image_for_seed_growth.png', img_for_seed)
    cv2.imshow('Mask image', img_for_seed)
    cv2.waitKey(0)



'''clicks = []

# Apply region growing to the 25% max binary image
def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1
    
    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))
    
    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))
    
    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    return out


# Region Growing Function
def region_growing(img, seed):
    seed_points = []
    outimg = np.zeros_like(img)
    seed_points.append((seed[0], seed[1]))
    processed = []
    while(len(seed_points) > 0):
        pix = seed_points[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            if img[coord[0], coord[1]] != 0:
                outimg[coord[0], coord[1]] = 255
                if not coord in processed:
                    seed_points.append(coord)
                
                processed.append(coord)
        seed_points.pop(0)
        
    return outimg

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), img_for_seed[y,x])
        clicks.append((y,x))
        
clicks = []




cv2.namedWindow('Select Seeds')
cv2.setMouseCallback('Select Seeds', on_mouse, 0, )

############################################################################



if args.method == 'colour':
    # Print time until region growing
    time_until_region_growing = cv2.getTickCount()
    time_until_region_growing = (time_until_region_growing - time_threshold_selection)/cv2.getTickFrequency()
    logging.info('Time until seed selection window since threshold selection: {} s'.format(time_until_region_growing))

else:
    time_until_region_growing = cv2.getTickCount()
    time_until_region_growing = (time_until_region_growing - time_after_select_roi)/cv2.getTickFrequency()
    logging.info('Time until region growing window displayed: {} s'.format(time_until_region_growing))

##########################################################################


cv2.imshow('Select Seeds', img_for_seed)









cv2.waitKey()

# Start timer after seed selection 
time_after_seed_selection = cv2.getTickCount()

seed = clicks[-1]

# Binary Seed Growth image
seed_growth_image = region_growing(img_for_seed, seed)

cv2.imwrite('image_after_seed_growth.png', seed_growth_image)


#cv2.imshow('Region Growing', seed_growth_image)
#cv2.waitKey(0)

# If there are more than one clicks
if len(clicks) > 1:

# For each additional click grow the corresponding seed
    for click in clicks[0:]:
        region = region_growing(img_for_seed, click)
        seed_growth_image = cv2.add(seed_growth_image, region)

# Display the mask
    #cv2.imshow('Region Growing', seed_growth_image)
    #cv2.waitKey(0)

# Find Product of Seed Growth image with grey ROI (i.e. grayscale seed growth image)
seed_growth_grey = cv2.bitwise_and(grey_image, seed_growth_image)
cv2.imwrite('seed_growth_image_multiplied_grey.png', seed_growth_grey)

#cv2.imshow('Grey Seed Growth Region', seed_growth_grey)
#cv2.waitKey(0)
time_seed_growth_complete = cv2.getTickCount()
time_seed_growth_complete = (time_seed_growth_complete - time_after_seed_selection)/cv2.getTickFrequency()
logging.info('Time of seed growth complettion: {} s'.format(time_seed_growth_complete))'''

##########################################################################








############################################################################




# Find frame width and height. Use .shape. For the moment process only one frame
# We convert the resolutions from float to integer.
#frame_height, frame_width = image.shape[:2]




# Calibrated width 
calibration_factor_w = calibrate_width(distance, frame_width)




# Calibrated height
calibration_factor_h = calibrate_height(distance, frame_height)




# Calibration factor for area
calibration_factor_a = calibrate_area(calibration_factor_w, calibration_factor_h)




# Apply contours to cropped_histogram_equalised_product_image to generate bounding box and display area
masked_frame = bounding_box(src= image, mask= img_for_seed, area_calibration= calibration_factor_a, width_calibration= calibration_factor_w, height_calibration= calibration_factor_h, kernel_size= 3, iterations= 1)
cv2.imwrite('masked_frame_w_bb.png', masked_frame)

#cv2.imshow('Bounding box', masked_frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



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
    # Write to video when called
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    # Load Video Stream
    video_cap = cv2.VideoCapture(args.video)

    # Frame counter


    count = 0
    # Process each frame separately

    for frame in frames_list:
        try:

        # Read frame
            ret, masked_frame = video_cap.read()
            count += 1

        # Apply bounding box to frame 
            masked_frame = bounding_box(src=masked_frame, mask=mask, area_calibration= area_calibration, width_calibration= width_calibration, height_calibration= height_calibration, kernel_size= 3, iterations=1)

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

write_masked_video(frames_list= frames, mask= img_for_seed, area_calibration= calibration_factor_a, width_calibration= calibration_factor_w, height_calibration= calibration_factor_h)


logging.info('Area Calibration Factor: {}'.format(calibration_factor_a))
logging.info('Width Calibration Factor: {}'.format(calibration_factor_w))
logging.info('Height Calibration Factor: {}'.format(calibration_factor_h))
logging.info('Frame Width: {}'.format(frame_width))
logging.info('Frame Height: {}'.format(frame_height))









