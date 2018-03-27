# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_examples/ahead_only.jpg "Traffic Sign 1"
[image5]: ./test_examples/double_curve.jpg "Traffic Sign 2"
[image6]: ./test_examples/no_entry.jpg "Traffic Sign 3"
[image7]: ./test_examples/speed_limit_20kmh.jpg "Traffic Sign 4"
[image8]: ./test_examples/turn_right_ahead.jpeg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kingxueyuf/self-driving-car/blob/master/project_2_traffic_sign_classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided NOT to convert the images to grayscale because I want to keep the channel so that the model could learn more

Here is an example of a traffic sign image with/without grayscaling (RGB).

![alt text][image2]

As a last step, I decided NOT to normalize the image data because the conv net could handle it

Here is an example of an original image and an augmented image:

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x12 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 24x24x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x16 				|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 10x10x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x18 					|
| Flatten				| outputs 1x450									|
| Fully connected		| outputs 1x240									|
| Batch Normalization	| 												|
| ReLU					|												|
| Dropout				|												|
| Fully connected		| outputs 1x120									|
| Batch Normalization	| 												|
| ReLU					|												|
| Dropout				|												|
| Fully connected		| outputs 1x84									|
| Batch Normalization	| 												|
| ReLU					|												|
| Dropout				|												|
| Fully connected		| outputs 1x43									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an BATCH_SIZE = 100, EPOCHS = 100, learning rate = 0.0001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.953 
* test set accuracy of 0.946

If an iterative approach was chosen:
What was the first architecture that was tried and why was it chosen?
* AlexNet.

What were some problems with the initial architecture?
* No batch normalization.
* No dropout.
* Too less layer.

How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

* Add more convolution layers.
* Add batch normalization to avoid negative effect of weight initialization.
* Add dropout to avoid overfitting.

Which parameters were tuned? How were they adjusted and why?

* learning rate, from 1e-4 to 1e-3.
* decay.

What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

* Convolution layer will learn the image features from different levels.
* Dropout layer will randomly drop neurals during training and uses all neurals mulitply by a factor during inference, by this way each neural will learn more information during training and leverage each of the neural during inference.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="./test_examples/ahead_only.jpg" width="100">
<img src="./test_examples/double_curve.jpg" width="100">
<img src="./test_examples/no_entry.jpg" width="100">
<img src="./test_examples/speed_limit_20kmh.jpg" width="100">
<img src="./test_examples/turn_right_ahead.jpeg" width="100">

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Keep left   									| 
| Speed limit (20km:h)	| Stop 											|
| Turn right ahead		| Turn right ahead								|
| No entry	      		| No entry						 				|
| Double curve			| No passing for vehicles over 3.5 metric tons  |


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the 1st image, the model is relatively sure that this is a Keep left sign (probability of 0.672), and the image is contain a Ahead Only sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6.7291129e-01         | Keep left  									| 
| 3.0129230e-01			| Speed limit (30km/h)							|
| 1.4237531e-02			| Speed limit (80km/h)							|

For the 2nd image, the model is relatively sure that this is a Stop sign (probability of 0.769), and the image is contain a Speed limit (20km:h) road sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.6970470e-01         | Stop  										| 
| 1.7717081e-01			| Speed limit (30km/h)							|
| 2.5567546e-02			| Bumpy road									|

For the 3rd image, the model is extremly sure that this is a Turn right ahead sign (probability of 0.995), and the image does contain a Turn right ahead sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.9514341e-01         | Turn right ahead   							| 
| 4.8565832e-03			| Ahead only									|
| 5.1939925e-10			| Keep left										|

For the 4th image, the model is extremly sure that this is a No entry sign (probability of 1.00), and the image is contain a No entry sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00         | No entry  									| 
| 1.5828524e-30			| Stop											|
| 2.2187330e-37			| Speed limit (20km/h)							|

For the 5th image, the model is extremly sure that this is a Priority road sign (probability of 1.00), and the image is contain a Double curve sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00         | No entry  									| 
| 7.0319961e-10			| Right-of-way at the next intersection			|
| 5.0769810e-13			| Traffic signals								|
 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


