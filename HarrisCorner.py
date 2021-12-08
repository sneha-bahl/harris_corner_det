"""
Harris Corner Detector

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01
#   ADDING THE K VALUE, USUAL RANGE (0.04,0.06); THE LARGER THE VALUE, 
#   THE MORE IT REFINES THE SELECTION OF POINTS; BUT TAKING IT TO BE 
#   TOO HIGH COULD ALSO RISK MISSING SOME CRITICAL POINTS
k = 0.04

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) # Prewitt mask
dy = dx.transpose()

image = cv2.imread('test1.jpg')
#   DISPLAY ORIGINAL IMAGE
cv2.imshow('Original',image)
bw = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

#   g GIVES THE GAUSSIAN WINDOW FUCNTION, THE PARAMAETERS SENT TO fspecial ARE THE WINDOW SIZE OF 
#   THE GAUSSIAN WINDOW (CREATED BY, fspecial) AND RETURNED TO g
g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
#   Compute the Harris Cornerness
######################################################################

#   WE CAN USE THE DIRECT FORMULAS TO DETERMINE THE DETERMINANT AND TRACE OF THE 'M' MATRIX
det  = Ix2 * Iy2 - Ixy * Ixy
trace = Ix2 + Iy2
#   THE FORMULA FOR CALCULATING RESPONSE
R = det - k* trace*trace
#print(R.shape)
#   DISPLAY RESPONSE IMAGE
cv2.imshow('R', R)
#print(R)
#print(np.amax(R))

######################################################################
#       Performing non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################

#   APPLYING THRESHOLDING
R_thresh = np.ones(R.shape)
for i, values in enumerate(R):
    for j, r in enumerate(values):
        #   WOULD USE ONLY >thresh IF THE PIXEL VALUES WERE INITIALLY IN THE RANGE [0,1] 
        #   BUT PIXEL VALUES ARE ACTUALLY IN [0,255] RANGE, SO THIS IS A BETTER WAY TO SET 
        #   THE THRESHOLD; THIS WAY IT CAN ALSO VARY ACCORDING TO THE IMAGE
        if r > (thresh*np.amax(R)):
            #   this will be a corner
            R_thresh[i, j] = r

cv2.imshow('R after thresh', R_thresh)
#print(R_thresh.shape)

#   USING A SLIDING WINDOW OVER THE R_THRESH TO FIND THE LOCAL MAXIMAS
kernel = np.zeros((13,13))
rows, cols = R_thresh.shape
ker_r , ker_c = kernel.shape
x_r = ker_r//2
x_c = ker_c//2
#   ASSIGNING THE CENTER VALUE TO 1 AND OTHERS ARE 0
kernel[x_r,x_c] = 1
#   ADDING PADDING TO IMAGE
padded_img = np.zeros(((rows + 2*x_r), (cols + 2*x_c)))
padded_img[x_r: padded_img.shape[0]- x_r, x_c: padded_img.shape[1]- x_c] = R_thresh
img1 = padded_img
output_image = np.zeros(img1.shape)
#   FINDING MAXIMA
for i in range(x_r, img1.shape[0]-x_r):
    for j in range(x_c, img1.shape[1]-x_c):
        #   GETTING THE VALUES FROM THE R_thresh OCCURING IN THE WINDOW 
        val = img1[i-x_r:i- x_r+ker_r , j-x_c: j-x_c+ker_c]
        #print(val)
        max = np.amax(val)
        #   IF MAXIMUM AMONGST THE NEIGHBORHOOD, SAVE THE VALUE TO THE output_image(CREATING A NEW IMAGE TO FIND ALL MAXIMAS)
        #   AND ASSIGN ALL THE NEIGHBOURING POINTS = 0
        if(img1[i,j] == max):
            output_image[i-x_r:i- x_r+ker_r , j-x_c: j-x_c+ker_c] = np.multiply(val, kernel)

R_maximum = output_image[x_r:-x_r, x_c:-x_c] #  IMAGE PART EXCLUDING THE PADDING
cv2.imshow('R_max', R_maximum)
#print(R_maximum)

#   CREATING A LIST OF ALL THE PIXEL LOCATIONS FOR THE CORNERS
corners = []
for i, values in enumerate(R_maximum):
    for j, r in enumerate(values):
        if(r != 0):
            corners.append((i,j))
#   PLOTTING THE CORNERS ON THE IMAGE
corn = np.asarray(corners)
#print(corn.shape)
final = plt.imread('test1.jpg')
plt.imshow(final, cmap='gray')
plt.scatter(corn[:,1],corn[:,0],c='r',s=3)
plt.title('Harris Corner')
plt.axis('off')

#   USING THE OPENCV IN-BUILT CORNER HARRIS DETECTOR
#   PARAMETERS-     (greyscale image, derivative blocksize, aperture of sobel operator, k )
#   USING SOME IDEA FROM:- REFERENCE: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
dst = cv2.cornerHarris(bw,3,13,0.04)
#   APPLYING THRESHOLD ( MAY VARY FOR DIFFERENT IMAGES), AND MARKING THE CORNERS IN THE IMAGE
image[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('In-built',image)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
