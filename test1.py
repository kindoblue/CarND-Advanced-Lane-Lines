import cv2
import numpy as np
import sys

this = sys.modules[__name__]

min, max = 0, 0
image = np.empty((720, 1280, 3))
window_name = ''

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)

    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[
        (scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    # Return the result
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output



def redraw():
    # bin = abs_sobel_thresh(this.image, 'y', 3, this.min, this.max)
    bin = mag_thresh(this.image, 3, (this.min, this.max))

    # convert from gray (one channel) to 3 channels, to have the
    # hstack working
    imgc = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

    both = np.hstack((image, imgc))

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


def SimpleTrackbar():

    # create the window
    cv2.namedWindow(window_name)

    # callback and trackbar
    cv2.createTrackbar('min', window_name, 0, 255, set_min)
    cv2.createTrackbar('max', window_name, 0, 255, set_max)

    # initialise
    redraw()

    while True:
        # if you press "ESC", it will return value 27
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    image = cv2.imread('images/test2.jpg')
    window_name = 'test'
    SimpleTrackbar()