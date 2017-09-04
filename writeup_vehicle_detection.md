##README

---

**Vehicle Detection Project**

## Structure of the repository
```angular2html
|-vehicle_detection
|---car_finder.py                     code for find cars in a video frame, and for processing video stream
|---svm_car_classify.py               code for training svm classifier to differentiate car images vs non car images
|---image_features.py                 code to construct features for the svm classifier
|-resources                           image and video resources
|---images                            image resources
|-----output_images                   output images
|-----test_images                     input images
|---videos                            videos resources
|-----input_videos                    input videos
|-----output_videos                   output videos
```

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./resources/images/output_images/hog.jpg
[image2]: ./resources/images/output_images/test6_car_finder.jpg
[image3]: ./resources/images/output_images/test2_car_finder.jpg
[image4]: ./resources/images/output_images/test3_car_finder.jpg
[image5]: ./resources/images/output_images/test5_car_finder.jpg
[video1]: ./resources/videos/output_videos/project_video_vehicle_detected.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images. 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG features of car and not car][image1]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters to find the ones with the good test accuracy and the low training time. Here are 

##### optimize pix_per_cell
|pix_per_cell|cell_per_block|time to extract HOG features (seconds)| features|time to train linear SVC| accuracy|time to predit 10 labels (seconds)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|16|4|36.7|576|1.29|96.22%|1.12e-3|
|8|2|55.92|7056|3.29|98.25%|8.1e-4|
|4|1|115.46|9216|35.29|97.68%|8.4e-4|
`pixel_per_cell = 8` and `cell_per_block = 2` are used

#### optimize color-space
|color-space|time to extract HOG features (seconds)|time to train linear SVC| accuracy|time to predit 10 labels (seconds)|
|:---:|:---:|:---:|:---:|:---:|
|YCrCb|55.92|3.29|98.22%|8.1e-4|
|LUV|59.06|20.16|97.46%|8.4e-4|
|YUV|56.6|2.8|98.34%|1.13e-3|
|HSV|56.74|4.36|98.14%|7.9e-4|
|HLS|56.73|5.72|98.31%|1.1e-3|
|RGB|55.58|21.36|969%|7.9e-4|
`color-space = YCrCb` is used

#### optimize orient
|orient|time to extract HOG features (seconds)|time to train linear SVC| accuracy|time to predit 10 labels (seconds)|
|:---:|:---:|:---:|:---:|:---:|
|16|61.62|3.18|98.17%|8.5e-4|
|12|55.92|3.29|98.25%|8.1e-4|
|8|53.48|10.36|97.94%|7.8e-4|

`orient = 12` is used 

The selected parameters can be seen in Line 39-43 of `config.py`. The parameters are configured at a central location `config.py`, so we can makes sure the same parameters are used for training, testing and sliding window search for cars in video processing.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with C = 1.0 (Line 106 of `vehicle_detection/svm_car_classify.py`). I tried `GridSearchCV` to optimize the kernel and C of the SVM, and find the best combination is `kernel='rbf', C=10`. However, the difference between the rbf kernel and linear kernel is not big (98.68% vs 98.9%), rbf kernel is much slower in training (2.8 vs 258 seconds) and testing (8.1e-4 vs 0.17897 seconds), and might have overfitting problem, and not shows better performance in video processing, so I decide to just use linear SVM instead. For linear SVM, I tested `C=0.1` and `C=10`, both results in a lower accuracy (98.51% and 98.39%) than using `C=1.0`. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search in the file `vehicle_detection/car_finder.py`. Four scales were used, as seen in `car_multiple_detections()` function of the file, as well as the table below. The right panel on the first row of the figure below plots the first two and last two windows of each scale, so we can visualize the start, end, size and overlap of each scale. Please notes the windows from a same scale were plotted used the same random color (also note the windows detected with car plotted on the left panel of the second role are plotting with the same corresponding colors as in the right panel of the first row).
I always use 75% overlap, so the windows form nice coverage, but not super tight. For each scale, ystart is always 400, which is slightly higher than the horizon; different values were chosen for yend: the larger the scale, the larger the yend.


|ystart|yend|scale|
|:---:|:---:|:---:|
|400|500|1.2|
|400|656|1.5|
|400|720|2.0|
|400|720|2.5|


![alt text][image2]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features, which provided a nice result.  Here are some 
more example images:

![alt text][image3]

![alt text][image4]

![alt text][image5]

I already described above how I optimize the performance of the SVM classifier.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./resources/videos/output_videos/project_video_vehicle_detected.mp4)



####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


Taking the figures above as example, the heatmaps are shown as the right panel of second row of each figure, the output of `scipy.ndimage.measurements.label()` on the integrated heatmap and the resulting bounding boxes are shown as the last row of each figure.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Despite extensive optimization I have done, the video processing is not 100% accurate. We can still see at a few spots, non car windows were recognized as car windows. One of the tip given was the car training data contains very similar figures of the same car, so it is better to manually separate training and test data to void overfitting. I did that in L61 -111 of `vehicle_detection\svm_car_classify.py`, which slightly reduce the test accuracy, but does not solve the problem.
There might be several limitation of this method: 1. the HOG features are not good enough to differentiate cars and non cars, 2. the training data set is too small, 3. SVM might not be a good enough for this problem. To make it more robust, I would like to include other features, more training data, as well as using other models, such as deep neural networks.


