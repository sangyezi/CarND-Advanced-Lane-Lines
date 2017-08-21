## Advanced Lane Finding Project

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

[image1]: resources/images/camera_cal_corners/calibration2_contrast.jpg "Undistorted"
[image2]: resources/images/output_images/test4_undist.jpg "Road Transformed"
[image3]: resources/images/output_images/test4_threshold.jpg "Binary Example"
[image4]: resources/images/output_images/straight_lines1_warped.jpg  "Warp Example"
[image5]: resources/images/output_images/test4_threshold_line2.jpg "Fit Visual"
[image6]: resources/images/output_images/test4_line.jpg "Output"
[video1]: resources/videos/output_videos/project_video.mp4 "Project Video"
[video2]: resources/videos/output_videos/challenge_video.mp4 "Challenge Video"

## Structure of the repository
```angular2html
|-camera_calibration                  codes for camera calibration
|---camera_calibration.py             generate camera matrix and distortion coefficient 
|---undistort_image.py*               undistort image
|---camera_data.p                     camera data generated by camera_calibration.py 
|---unwarp_chessboard.py              transform perspective of chessboard images
|-line_finder                         codes for lane line identification pipeline
|---module_gui.py                     GUI for turning channel, gradient and threshold of image filtering
|---warp_matrix.p                     perspective matrix, generated by generate_perspective_matrix.py
|---workflow.py*                      the pipeline to identify lane lines in image frames of a video
|---generate_perspective_matrix.py    generate perspective matrix to warp and unwarp an image
|---thresholding.py*                  filters (or thresholding) images
|---locate_lane_lines.py*             locate lane lines on a thresholded and transformed image
|---transform_perspective.py*         transform perspective of a image
|-config.py                           config input and output resource folders
|-resources                           image and video resources
|---images                            image resources
|-----camera_cal                      input images for camera calibration
|-----camera_cal_corners              output images from camera calibration
|-----output_images                   output images
|-----test_images                     input images
|-----video_frames                    to store frames images of videos generated by video_extracted.py
|---videos                            videos resources
|-----input_videos                    input videos
|-----output_videos                   output videos
|-utils                               codes as utility
|---video_extract.py                  transform video to image frames
```
python files marked by `*` compose the pipeline for video processing, other python files are calibration or util files.
Please refer to documentations in the codes for more details

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `camera_calibration/camera_calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_point` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (funciton `threshold_pipeline()` in `line_finder/thresholding.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes in functions `main()` and `generate_matrix()` in the file `line_finder/generate_perspective_matrix.py` The `generate_matrix()` function takes as inputs source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points:

```python
src = np.float32([[195, height], [593, 450], [689, 450], [1125, height]])
dst = np.float32([[315, height], [315, 0], [965, 0], [965, height]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 195, 720      | 315, 720      |
| 593, 450      | 315, 0        | 
| 689, 450      | 965, 0        |
| 1125, 720     | 965, 720      |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

The `perspective matrix` and `inverse perspective matrix` were stored in pickle files. A class called `TransformPerspective` in the file `line_finder/transform_perspective.py` reads them back from the pickle file and applies the perspective transformation for the lane identfication pipeline.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #268 through #273 in my code in `line_finder/workflow.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `Lane.visualize()` in `line_finder/workflow.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


Here's a [my project video result](resources/videos/output_videos/project_video.mp4), or view [here](https://www.youtube.com/watch?v=lfTVZt_4gDk&feature=youtu.be)


Here's a [my challenge video result](resources/videos/output_videos/challenge_video.mp4), or view [here](https://www.youtube.com/watch?v=TVHDvATxmCE&feature=youtu.be)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


