import cv2
import numpy as np
import sys

this = sys.modules[__name__]

min, max, min1, max1, min2, max2 = 0, 0, 0, 0, 0, 0
image = np.empty((720, 1280, 3))
window_name = ''

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)

    # here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[
        (scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    # Return the result
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # convert to 8bit int
    absgraddir = absgraddir.astype(np.uint8)

    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    # return the binary image
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output


def pipeline(img, s_thresh=(115, 255), l_thresh=(25, 100), sx_thresh=(25, 100)):

    # copy the input image to avoid changing it
    img = np.copy(img)

    # convert to HLS color space and separate the lightness
    # and saturation channels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    # apply sobel filter in the x direction
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)

    # get the absolute value because also negative gradients represents
    # vertical changes
    abs_sobelx = np.absolute(sobelx)

    # redistribute values on the entire 0..255 range
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # threshold the x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[
        (scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255

    # threshold the saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

    # convert to 8 bit
    s_binary = s_binary.astype(np.uint8)

    # threshold the lightness channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 255
    l_binary = l_binary.astype(np.uint8)

    # stack each channel for debug
    colors = np.dstack((sxbinary, s_binary, l_binary))

    # convert to binary image where the lightess AND saturation bits have to be
    # both present to contribute, together with the gradient on x axes
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 255) & (s_binary == 255) | (sxbinary == 255))] = 255
    binary = np.dstack((binary, binary, binary))

    # return the images
    return colors, binary

def redraw():

    '''
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    
    thresh = (100, 255)
    bin = np.zeros_like(S)
    bin[(S > thresh[0]) & (S <= thresh[1])] = 255
    '''

    '''
    bin = abs_sobel_thresh(this.image, 'y', 3, this.min, this.max)
    fmin = min/1000.0
    fmax = max/1000.0
    bin = dir_threshold(this.image, 3, (fmin, fmax))
    '''

    s_thresh = (this.min, this.max)
    sx_thresh = (this.min1, this.max1)
    l_thresh = (this.min2, this.max2)

    _, binary = pipeline(image, s_thresh=s_thresh, l_thresh=l_thresh, sx_thresh=sx_thresh)

    # convert from gray (one channel) to 3 channels, to have the
    # hstack working
    #imgc = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

    both = np.hstack((image, binary))

    # show in window
    cv2.imshow(window_name, both)


# callback function
def set_max(value):
    this.max = value
    redraw()


# callback function
def set_min(value):
    this.min = value
    redraw()


def set_max1(value):
    this.max1 = value
    redraw()


# callback function
def set_min1(value):
    this.min1 = value
    redraw()


def set_max2(value):
    this.max2 = value
    redraw()


# callback function
def set_min2(value):
    this.min2 = value
    redraw()


def SimpleTrackbar():

    # create the window
    cv2.namedWindow(window_name)

    # callback and trackbar
    cv2.createTrackbar('sat min', window_name, 0, 255, set_min)
    cv2.createTrackbar('sat max', window_name, 0, 255, set_max)
    cv2.createTrackbar('grad min', window_name, 0, 255, set_min1)
    cv2.createTrackbar('grad max', window_name, 0, 255, set_max1)
    cv2.createTrackbar('light min', window_name, 0, 255, set_min2)
    cv2.createTrackbar('light max', window_name, 0, 255, set_max2)

    # initialise
    redraw()

    while True:
        # if you press "ESC", it will return value 27
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    image = cv2.imread('images/test5.jpg')
    window_name = 'test'
    SimpleTrackbar()