# **Behavioral Cloning** 

This writeup contains the approach taken to drive the car autonomously in the simulator.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/network.jpg "Model Visualization"
[image2]: ./writeup_images/center_image.jpg "Image from center camera"
[image3]: ./writeup_images/left.jpg "Left Recovery Image"
[image4]: ./writeup_images/center.jpg "Center Recovery Image"
[image5]: ./writeup_images/right.jpg "Right Recovery Image"
[image6]: ./writeup_images/center_image.jpg "Normal Image"
[image7]: ./writeup_images/flipped_image.jpg "Flipped Image"

## [Rubric Points:](https://review.udacity.com/#!/rubrics/432/view)
**I. Files Submitted & Code Quality**
   
1. Submission includes all required files and can be used to run the simulator in autonomous mode.
2. Submission includes functional code.
3. Submission includes functional code.

**II. Model Architecture and Training Strategy**

1. An appropriate model architecture has been employed.
2. Attempts to reduce overfitting in the model.
3. Model parameter tuning.
4. Appropriate training data.

**III. Architecture and Training Documentation**  

1. Solution Design Approach
2. Final Model Architecture
3. Creation of the Training Set & Training Process

---

### I. Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `model.ipynb` containing the script to create and train the model
* `model.h5` containing a trained convolution neural network 
* `drive.py` for driving the car in autonomous mode
* `video.py` for saving the video from front camera in autonomous mode.
* `writeup_report.md` which summarizing the results
* `video.mp4` shows video of one lap of the car running in autonomous mode in the simulator.(Camera View)
* `Behavioral Cloning.mp4` shows video of one lap of the car running in autonomous mode in the simulator.(3rd person view)
* `writeup_images` contains images used in this writeup.
* `IMP links for v2.txt` contains links I have to refer for version 2 of this project.

#### 2. Submission includes functional code
Using the Udacity provided [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip) and my `drive.py` file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.ipynb` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### II. Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

- I have used Nvidia's end-to-end model.
- This model consists of following layers:
	- Normalization Layer - 1
	- Convolutional Layers - 5
	- Fully connected Layers - 3
	- Output Layer - 1
- Normalization Layer: The data to be trained on is normalized using Keras Lambda Layer. 
- The model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64
- The model includes RELU layers to introduce nonlinearity.
- 3 fully connected layers gives a output feature array.
- The output layer gives the steering value for the corresponding image.

#### 2. Attempts to reduce overfitting in the model

- Loss on test and validation set was calculated using 'Mean Squared Error'. Comparing the values, I came to know whether the model is overfitting or not.
- Also the model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

- Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
- I have also used data from cameras attached on the left and right side of the car.
- Further, data augmentation is done to generate more data for training the model and make it unbiased.

---

### III. Architecture and Training Documentation

#### 1. Solution Design Approach

- The overall strategy for deriving a model architecture was to get an appropriate data for training. Since the machine learning algorithms works on *garbage input, garbage output* principle, using a good data is the key for this project.

- I wasn't sure about the uniformity of the data if I collected it by myself in simulator so I did data augmentation to increase the quantity of the data.

- My first approach was to use a Nvidia's end-to-end convolution neural network model. I thought this model might be appropriate because the engineers at Nvidia
have already validated this model by driving a car autonomously in real world. (Also getting hands on such a practically applied model would be amazing!)

- For data preparation, I used the data from all three cameras, added correction factor to the left and right camera image, and augmented the data .

- In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

- I trained the model on this data and now the final step was to run the simulator to see how well the car was driving around track one. 
- There were a few spots where the vehicle fell off the track like at the end of the bridge and the curve after the bridge.
- To improve the driving behavior in these cases, I tuned the correction factor to 0.10 and changed the color-space of all the images from BGR to RGB respectively.

- At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers:

- Normalization Layer - 1
- Convolutional Layers - 5
- Fully connected Layers - 3
- Output Layer - 1

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back in the center of the road. Following are the images from 3 cameras used to train the model to recover:

**Left Camera:**  
![alt text][image3]

**Center Camera:**   
![alt text][image4]

**Right Camera:**   
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help the model to turn steering on both the sides. For example, here is an image that has then been flipped:

![alt text][image6] ![alt text][image7]


After the augmentation process, I had **48,216** number of data points. I then preprocessed this data by normalizing data and mean centering the data.

I finally randomly shuffled the data set and put **20%** of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was **5** as evidenced by test loss and validation loss. I used an **adam optimizer** so that manually training the learning rate wasn't necessary.

---

[Video of the Project](./video.mp4)

---