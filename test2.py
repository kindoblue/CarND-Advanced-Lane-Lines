import numpy as np
import cv2
import pickle
from moviepy.editor import VideoFileClip
from functools import partial


def binary(img, s_thresh=(115, 255), l_thresh=(84, 255), sx_thresh=(25, 100)):
    """This function get an input image and return a binary images with
       lane lines detected. 
       The saturation, lightness, gradient thresholds can be customized 
       for experimentation. The default values are the one I found the best 
       choice for the images at hand"""

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

    # convert to binary image where the lightess AND saturation bits have to be
    # both present to contribute, together with the gradient on x axes
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 255) & (s_binary == 255) | (sxbinary == 255))] = 255
    binary = np.dstack((binary, binary, binary))

    # return the images
    return binary


def change_perspective(corners):
    """Change perspective on the input image, that has to be 
    undistorted. The corners are sent in the following order:
    bottom left, top left, top right, bottom right"""

    # create the source region
    src = np.float32(corners)

    # unpack the corners
    bl, tl, tr, br = corners

    # adjust top left and top right corner to
    # form a rectangle, to be used as destination
    tl = (bl[0], 0)
    tr = (br[0], 0)

    # create the destination rectangle
    dst = np.float32([bl, tl, tr, br])

    # calculate the matrix to transform from src to dst
    M = cv2.getPerspectiveTransform(src, dst)

    # and from dst to src
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


def slide_window(image, startx, nonzerox=None, nonzeroy=None, nwindows=9):
    """It slide a window from the bottom, with startx x coordinate.
    It returns the indices of the points belonging to the line"""

    if nonzerox is None:
        # x, y of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

    # calculate the height of the windows
    window_height = np.int(image.shape[0] / nwindows)

    # current positions to be updated for each window
    currentx = startx

    # set the width of the windows +/- margin
    margin = 100

    # set minimum number of pixels found to recenter window
    minpix = 50

    # create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # step through the windows one by one
    for window in range(nwindows):

        # calculate window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_x_low = currentx - margin
        win_x_high = currentx + margin

        # identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) &
                     (nonzeroy < win_y_high) &
                     (nonzerox >= win_x_low) &
                     (nonzerox < win_x_high)).nonzero()[0]

        # append these indices to the lists
        lane_inds.append(good_inds)

        # if you found > minpix pixels, re-center next window
        # on their mean position
        if len(good_inds) > minpix:
            currentx = np.int(np.mean(nonzerox[good_inds]))

    # concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    return lane_inds


def find_lines(image, p_left_fit=None, p_right_fit=None):
    """Returns the lane lines as function fit, given a binary image. It uses
    the previous fit to speed up the calculations"""

    # if it a 3 channel binary image, make it one channel
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # x, y of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if p_left_fit is None:
        # no previous frame where to start, let's use the sliding
        # window method

        # take a histogram of the bottom half of the image
        histogram = np.sum(image[int(image.shape[0] / 2):, :], axis=0)

        # find the peak of the left and right halves of the histogram
        # these will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        left_lane_inds = slide_window(image, leftx_base, nonzerox, nonzeroy)
        right_lane_inds = slide_window(image, rightx_base, nonzerox, nonzeroy)

    else:
        # we have a fit from the previous frame, so we use it to calculate
        # the points in the +/- margin neighborhood
        margin = 100

        left_lane_inds = ((nonzerox > (p_left_fit[0] * (nonzeroy ** 2) + p_left_fit[1] * nonzeroy + p_left_fit[2] - margin))
                          & (nonzerox < (p_left_fit[0] * (nonzeroy ** 2) + p_left_fit[1] * nonzeroy + p_left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (p_right_fit[0] * (nonzeroy ** 2) + p_right_fit[1] * nonzeroy + p_right_fit[2] - margin)) &
                           (nonzerox < (p_right_fit[0] * (nonzeroy ** 2) + p_right_fit[1] * nonzeroy + p_right_fit[2] + margin)))

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(left_fit=None, right_fit=None)
def _pipeline(img, cmx, dist):

    # un-distort the frame using camera matrix and distortion coeffs
    undistort_img = cv2.undistort(img, cmx, dist, None, cmx)

    # get the binary image
    bin_img = binary(undistort_img)

    # define the trapezoid which encloses the road ahead
    corners = [(277, 670), (585, 457), (702, 457), (1028, 670)]

    # get the matrix to change perspective and the one to reverse
    M, Minv = change_perspective(corners)

    # image shape needed
    img_shape = (img.shape[1], img.shape[0])

    # warp the frame
    binary_warped = cv2.warpPerspective(bin_img, M, img_shape, flags=cv2.INTER_LINEAR)

    # search for the lines in the frame
    _pipeline.left_fit, _pipeline.right_fit = find_lines(binary_warped,
                                                         _pipeline.left_fit,
                                                         _pipeline.right_fit)

    # output image
    layer = np.zeros_like(binary_warped).astype(np.uint8)

    # generate x and y values for plotting
    ploty = np.linspace(binary_warped.shape[0] - 1, 0, 30)
    left_fitx = _pipeline.left_fit[0] * ploty ** 2 + _pipeline.left_fit[1] * ploty + _pipeline.left_fit[2]
    right_fitx = _pipeline.right_fit[0] * ploty ** 2 + _pipeline.right_fit[1] * ploty + _pipeline.right_fit[2]

    # zip x and y's to generate array of points
    left_pts = np.dstack((left_fitx, ploty))
    right_pts = np.flip(np.dstack((right_fitx, ploty)), axis=1)

    # create the polygon.
    polygon = np.array(np.concatenate((left_pts, right_pts), axis=1), dtype=np.int32)

    # draw the polygon in the warped space
    cv2.fillPoly(layer, polygon, (0, 255, 0))

    # unwarp and add to the original image
    layer_unwarp = cv2.warpPerspective(layer, Minv, img_shape, flags=cv2.INTER_LINEAR)

    return cv2.addWeighted(img, 1, layer_unwarp, 0.3, 0)


if __name__ == '__main__':

    # load camera matrix
    with open(r"cmx.p", "rb") as cmx_file:
        cmx = pickle.load(cmx_file)

    # load distortion coeffs
    with open(r"dist.p", "rb") as dist_file:
        dist = pickle.load(dist_file)

    # partial application on _pipeline function, so I can
    # send in the camera matrix and distortion coeffs without
    # relying on globals.
    pipeline = partial(_pipeline, cmx=cmx, dist=dist)

    # open the clip
    clip = VideoFileClip('project_video.mp4')

    # process the clip
    processed_clip = clip.fl_image(pipeline)

    # finally save the clip
    processed_clip.write_videofile("project_video.out.mp4", audio=False)
