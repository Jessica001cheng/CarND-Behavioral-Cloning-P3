# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/Normal_Image.jpg "Grayscaling"
[image3]: ./examples/Left_recovery.jpg "Recovery Image"
[image4]: ./examples/Left_recovery2.jpg "Recovery Image"
[image5]: ./examples/Right_recovery.jpg "Recovery Image"
[image6]: ./examples/Before_Augment.jpg "Normal Image"
[image7]: ./examples/Augment_image.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 86-24) 

The model includes RELU layers to introduce nonlinearity (code line 86), and the data is normalized in the model using a Keras lambda layer (code line 84). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 98,100,102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the nVidia model I thought this model might be appropriate because it is good to train images from single front-facing camera to steering command.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add dropout (with a keep probability of 0.8) between the three sets fully-connected layer. 

Then I use Adam optimizer was with default parameters and choose the loss function as mean squared error (MSE). The final layer is a fully-connected layer with a single neuron. [model.py lines 103-105]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track:

1. the 1st curve before the bridge.
2. the 2nd curve where with water on left side
3. the 3rd curve which is very sharp

to improve the driving behavior in these cases, I record more data on these areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

|Layer (type)               |  Output Shape           |   Param #   |
|---------------------------|:------------------------|:------------|
|cropping2d_1 (Cropping2D)  |  (None, 65, 320, 3)    |    0       |  
|lambda_1 (Lambda)          |  (None, 65, 320, 3)    |    0       |  
|conv2d_1 (Conv2D)          |  (None, 31, 158, 24)   |    1824    |  
|conv2d_2 (Conv2D)           | (None, 14, 77, 36)      |  21636  |   
|conv2d_3 (Conv2D)           | (None, 5, 37, 48)      |   43248  |   
|conv2d_4 (Conv2D)      |      (None, 3, 35, 64)      |   27712  |   
|conv2d_5 (Conv2D)       |     (None, 1, 33, 64)     |    36928  |   
|flatten_1 (Flatten)      |    (None, 2112)         |     0      |   
|dense_1 (Dense)           |   (None, 100)         |      211300 |   
|dropout_1 (Dropout)        |  (None, 100)        |       0      |   
|dense_2 (Dense)           |   (None, 50)        |        5050   |   
|dropout_2 (Dropout)       |   (None, 50)       |         0      |   
|dense_3 (Dense)           |   (None, 10)      |          510    |   
|dropout_3 (Dropout)       |   (None, 10)     |           0      |   
|dense_4 (Dense)           |   (None, 1)     |            11     |   

Total params: 348,219

Trainable params: 348,219

Non-trainable params: 0

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by comparing the training loss and validation loss with different epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
