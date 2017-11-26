# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output/car_noncar.png
[image2]: ./output/rgb_hog.png
[image3]: ./output/rgb_11_16_hog.png
[image4]: ./output/yuv_hog.png
[image5]: ./output/yuv_11_16_hog.png
[image6]: ./output/hog_transformation.png
[image7]: ./output/hog_features.png
[image8]: ./output/histogram.png
[image9]: ./output/spatial_features.png
[image10]: ./output/wrong_colorspace.png
[image11]: ./output/search_window_01.png
[image12]: ./output/search_window_02.png
[image13]: ./output/working_classify.png
[image14]: ./output/working_classify_2.png
[image15]: ./output/working_pipeline_01.png
[image16]: ./output/working_pipeline_02.png
[image17]: ./output/working_pipeline_03.png
[image18]: ./output/working_pipeline_04.png
[image19]: ./output/carposition_heatmap.png
[image20]: ./output/falsepositive.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it :smile:

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

The code for this step is contained in the file `Helper.py`. There you can find the method `get_hog_features` in the **line 89**.
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations` and `pixels_per_cell`).  Therefore, I tested this different combinations, not only to see the output that gaves me, but including a classify with those parameters, to check each one was the best one.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

| Images        | Time to extract features | Time to train the SVC | Accuracy		| Shape of features|
|:-------------:|:------------------------:|:---------------------:|---------------:|-----------------:|
| 17760		    | 60.62 seconds		       | 97.83 seconds		   | 0.971			| (17760, 5292)	   |

Now for `RGB` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image3]

| Images        | Time to extract features | Time to train the SVC | Accuracy		| Shape of features|
|:-------------:|:------------------------:|:---------------------:|---------------:|-----------------:|
| 17760		    | 51.64 seconds		       | 5.28 seconds		   | 0.96			| (17760, 1188)    |

---
In addition, I tried for the `YUV` color space. Therefore, here is the configuration, `YUV` with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image4]

| Images        | Time to extract features | Time to train the SVC | Accuracy		| Shape of features|
|:-------------:|:------------------------:|:---------------------:|---------------:|-----------------:|
| 17760		    | 72.33 seconds		       | 31.82 seconds		   | 0.98			| (17760, 5292)    |


`YUV` with HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image5]

| Images        | Time to extract features | Time to train the SVC | Accuracy		| Shape of features|
|:-------------:|:------------------------:|:---------------------:|---------------:|-----------------:|
| 17760		    | 45.71 seconds		       | 2.44 seconds		   | 0.9755			| (17760, 1188)    |


---
At the end I choose the `RGB` color space with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Even the `YUV` being fast, I combined the color features as well, so I decided to keep all in the same color space, that is `RGB`.

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the lessons as base for all I did in this project, therefore I used Support Vectors Classifiers to train my data.
Moreover, I add the color features, it took a while to train (about 7 minutes), however I got **98.96 %** of accuracy!
Here is all I have tried, and the final one that I select to keep going. All of that, I trained in the data that was provided by Udacity, a total of 17760 images of cars and non cars.

|Color Space|Orientation|Pixels |Block|Spatial|Histogram| Extract Features|Train        | Accuracy| Shape       |
|:---------:|:---------:|:-----:|:---:|:-----:|:------:|:---------------:|:-----------:|:-------:|:-----------:|
| RGB		| 9         | (8,8) |(2,2)|    -  |-        |60.62 seconds    |97.83 seconds|0.971    |(17760, 5292)|
| RGB		| 11        |(16,16)|(2,2)|    -  |-        |51.64 seconds    |5.28 seconds |0.96     |(17760, 1188)|
| YUV		| 9         |(8,8)  |(2,2)|    -  |-        |72.33 seconds    |31.82 seconds|0.98     |(17760, 5292)|
| YUV		| 11        |(16,16)|(2,2)|    -  |-        |45.71 seconds    | 2.44 seconds|0.9755   |(17760, 5292)|
| RGB		| 9         | (8,8) |(2,2)|   32  |32, range 0-255|427.04 seconds|106.78 seconds|0.9896|(17760, 8460)|

Here is a image of my final result of `HOG` in the channel red of the image.
![alt text][image6]

Moreover, here are the plot of its features.
![alt text][image7]

Also, here is the histogram of the image:
![alt text][image8]

To conclude, it's spatial features:
![alt text][image9]

---
I also decided to use RGB because I had a bad time reading the images, because **matplotlib** reads **png** in a range of 0 to 1. However, when it reads a **jpg** the scale is 0 to 255 :frowning: I did not noticed that until I try to training my data and predict on it :disappointed: Then I changed everything to RGB and start to use **cv2** to read the images.
![alt text][image10]

See in the image above that the colors of the boxes are black? That is because it messed up the scale of the images, so when it sum 255 in the blue channel, it get over the range and turn it black. Moreover, the predictions were terrible because of that. This image was ok, however the output video, was not.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I first tried with a simple search window criteria, all windows with the size of 128 by 128 pixels and an overlap of 0.85.
![alt text][image11]

However, this increased the time of extracting features of the cars far away was not well predicted.
Because of that, I change the windows for two sizes of windows. The first one 128 by 128 pixels and an overlap of 0.75 and a second one with 96 by 96 pixels and overlap of 0.75. Those windows sizes got a different region of the image, as is possible to see in the image below:
![alt text][image12]

|Color in the image above| Window Size   | Overlap | x limits | y limits |
|:----------------------:|:-------------:|:-------:|:--------:|:--------:|
|RED                     |96x96		     | 0.75    |(500,1300)|(380,572) |
|BLUE                    |128x128        | 0.75    |(500,1300)|(380,750) |


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some examples of my pipeline working.

|image| image|
|:----------------------:|:------------------:|
|![alt text][image13]    |![alt text][image14]|
|![alt text][image15]    |![alt text][image16]|
|![alt text][image17]    |![alt text][image18]|


To get the best of the classifier I used not only the HOG features, but I combined them with the spatial features and histogram features. To combine the features I used the `StandardScaler` provided in the `sklearn`. That gave a total of 8460 features values for each image. It increase the time to extract the features, train and predict, however it push up the accuracy to 98.96 %.
In addition, I used a heat map, plus labels of the library `scipy` to identify the regions where is most probable to find a car.
![alt text][image19]

The final version was really good. However, contrary to the ideal of a vehicle detection, does not have any false positive, it happens twice in the beginning of the video :disappointed:
![alt text][image20]


---

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)

Youtube [link](https://youtu.be/BtlonPwnouQ)!

Observatio: To compile this video, it took 25 minutes in a total of 1261 frames with a rate of 1.25 second per frame.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The pipeline works well for the near-fields vehicles. For that, I had to filter the false positives. Therefore, I did sum of the last 10 frames (file `Pipeline.py` start at line 109 and finish at 127), getting the heats of them. Then, I threshold on that sum to get the trust-able car identification and a smoother bound boxes for the car.
Moreover, to try to avoid the false positives setting a minimal area for the car, using the variable `self.data_to_be_printed.threshold_car_area` at `Pipeline.py` line 128. That take off smallest false positives, getting a better result over all.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- I had a lot of problem to read properly the images and use them in the classifier. Matplotlib getting the scale of png from 0 to 1, took me a lot of time.
- Elimite all the false positive is a hard thing to do!!!
- I am completely sure a video was taken in this same road with same cars, but raining, my vehicle detection would be gone crazy. The same for low light conditions, like a video during the night.
- To make more robust I believe that the SVM should be changed to a CNN. However, that is just a guess.
- Take more data to train the classifier, car and roads at night, rain and other conditions.

Moreover, a improvement of this project would be identify the speed and distance of the cars. That would improve a lot the useful of the project.
Well, like Sebastian said in one of the lessons of Self-Driving Car, most of the hard work of a self-driving car is the **perception**. I can understand it now! It is a hard work.

