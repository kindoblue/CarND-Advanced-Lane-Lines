## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/warped.png "Transformed"
[image3]: ./output_images/util.png "Utility"
[image4]: ./output_images/warped2.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/snapshot.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the notebook `part1_calibration.ipynb`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

In the same notebook I also test the perspective change, by reprojecting the corner of the chessboard to a plane parallel to the camera plane. Here's the result

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of lightness, saturation and gradient thresholds to generate a binary image. To decide about the most effective threshold values, I created an quick a dirty app, `test1.py`,  with sliders to adjust  the thresholds and see the change runtime. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `change_perspective()`, which appears in file `part2_test_pipeline.ipynb`.  This function takes as inputs an image (`img`), as well as corners of the trapezoid to project. The destination points are just calculated based on the source input. I chose the hardcode the destination points in the following manner:

```python
corners = [(190, 720), (589, 457), (698, 457), (1145, 720)]
```

This resulted in the following source and destination points:

| Source        | Destination  | 
|:-------------:|:------------:| 
| 190, 720      | 190, 720    | 
| 589, 457      | 589, 0      |
| 698, 457      | 698, 0      |
| 1145, 720     | 1145, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the notebook `part3_video_processing.ipynb` in the function `calculate_curvature_radius` (5th cell).  The code is basically the one from the lessons, not invented here. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The project was challenging because of all the moving parts. The most difficult stuff was all the errors I had to face due to shape mismatch in images and in arrays. For example, when creating an array of points for `cv2.polyfill`, I struggle to come up with working code (concatenate on the second axe otherwise you have two polygons, and stuff like that)
Another difficult part was to come up with a good binary image. At a certain point I decided to create an app with sliders to accelerate the experiments.

Unfortunately due to lack of time I could not tackle the challenges.     
