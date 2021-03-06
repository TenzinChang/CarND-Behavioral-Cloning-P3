# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (NOTE: i didn't modify original)
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

Overall I tried to:
- First setup a simple model, using nvidia as template.
- Augment the data, tried different augmentation
- Tune the model.

For model architecture I tried various strategies but end up basically re-using the nvidia architecture in the lecture video, and only added Dropout because it worked best. Different model architectures I've tried:
- Started with simply 1 dense layer, no convolutions. This doesn't work well but at least prove the car can drive.
- Stack several Cond2D layers, and try to add Dropout between them, however, I noticed Dropout between the Conv2D layers doesn't help the performance, so I removed them and only add Dropout in the fully connected layers, which did help.
- Tried various kernel sizes: (3x3, 5x5) instead of (2,2), because augmented data chopped off the top/bottom of image, i also try asymetric kernal sizes (2x3, 2x4), etc.. But these settings don't seem to help and I end up using the simplest one used in the lecturer's video.
- Tried various filter sizes: from 3->6->12->24->48->96, and different variations, but these tuning seem not able to perform better than the nvidia settings or only marginally, or simply caused validation errors to go up (overtrained).
- in my working copy of model.py I had many lines commented out after various tries, but unfortunately I "clean" them up before submission. I'll remember to keep them next time :)

For training:
- I augment the data first as the videos suggested: flip and image, also add left/right camera in addition to middle.
- add ploting for training/validation errors in each epoch, this way i can determine if the network overfit.
- I was using AWS/carnd instance, then the copy model.h5 back and forth between my laptop and AWS was too much, also my laptop was dying trying to run the model. So I decided to get a linux machine w/ GPU, it's almost impossible to get GPU now because of those crazy miners, anyhow, I managed to pay a premium $1100 for Nvidia 1080Ti, and took 2 weeks to setup Ubuntu 16.04 running carnd-term1 conda environment.yml w/ various updates of CUDA/cuDNN library because tensorflow has very specific version requirements(using CUDA 8.0, which depends on cnDNN 5.1). It was a lot of fun!

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network that copy basic nvidia architecture as discussed in the video.

The model includes RELU layers to introduce nonlinearity, and Dropout for regularization.


#### 2. Attempts to reduce overfitting in the model

I notice the validation error is not going down, so I add various Dropout layers to regularize.

#### 3. Model parameter tuning
I tried various tuning like adding more layers, use different size filters, add more/less dropout.

#### 4. Appropriate training data
To augment data, i use:
- normalize (per video instructions)
- flip the images
- add left/right images to training, and tried various correction factor (0.2 is better than none or other values).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to just keep it simple. I started w/ the simplest 1 layer as
David suggested, then add more layers to it.
My split of training/validation is 0.8/0.2

To combat the overfitting, I added Dropout to the various layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I played with different correction factor, finally arrived at 0.2.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road (except a few close calls)

#### 2. Final Model Architecture

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1            
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1              
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1             
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1          
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2          
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3        
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4           
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 33, 64)     0           convolution2d_5           
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           dropout_1                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           270464      flatten_1                 
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 128)           0           dense_1                   
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 64)            8256        dropout_2                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 64)            0           dense_2                   
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            1040        dropout_3                 
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             17          dense_3                
====================================================================================================
Total params: 411,125
Trainable params: 411,125


#### 3. Creation of the Training Set & Training Process

During the training process, I first tried to drive myself to create training data, but i was a terrible driver and had problem keeping the car in the lane, so I decided to use the data provided by the class.

Before I augment the data, I simply use the normalized images to train the model, which is simply stacked Conv2D layers and not much tuning, and it still drives very poorly, so I began to augment the data:
- flipping the images (left <-> right), which is big win.
- add left/right lane, and tried various correction factor (0.0, 0.1, 0.2, 0.4...), and finally settled w/ 0.2
- only after fully augmented the data, then I began to tune the model and tried different Dropout, filter, kernel size, etc.. (see above). There are probably 20+ iterations in total.

I created the run1.mp4 per instruction. Due to size limit of Github 32MB can't upload here, the link is below:
https://drive.google.com/open?id=1-aeKxF5ZR5cEfylHPUDqzGLKktr1PfGy
