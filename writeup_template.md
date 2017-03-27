#**Traffic Sign Recognition** 
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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
## Submitted files


## Data Set Summary & Exploration

The code for this step is contained in the cell #4 of the IPython notebook. As long as the dataset is loaded 
in from of numpy array, most summary points are immediately available:    

* The size of training set is 34799 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32x32 px with 3 color channels (RGB)
* The number of unique classes/labels in the data set is 43 (0-42)

The code in cell #5 visualizes a sample content of the training dataset. To better understand the challenge, I 
plot 10 randomly selected images for each label. This visualization shows how diverse the images belonging to the 
same class are. 

Later in the project I also explore how many samples of each class there are in training and validation dataset.   

## Data preprocessing 

As a preprocessing step, I convert images to grayscale (cell #7). It was mentioned in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
that grayscale images may show better performance. Indeed, my experiments in this project showed the prediction accuracy 
gain from ~0.87 to ~0.9 when I implemented this step.
   
I decided not to do any further preprocessing, because I managed to achieve required accuracy threshold of 0.93 with other techniques,
described below. 

## Model architecture

As a starting point, I used the LeNet-5 network architecture. The network without modifications showed prediction accuracy of 0.87 
on the validation dataset, which I managed to raise to 0.9 after converting the images to grayscale. 

When training the network, I noticed that the training error was decreasing to almost 0, but the validation error started to 
increase after a certain iteration, which signified that the model was overfitting. To prevent that, I decided to try using 
dropout in fully connected layers. Having experimented with different dropout rates, I found that the dropout rates of 0.3 through 0.6 were
giving the best results on the validation dataset. I decided to use the dropout rate of 0.5 eventually.  

## Final network architecture 

The code that constructs the final network is located in the cell #8. 
My final model consists of the following layers:

## Model training 

The code for training the model is located in cells #9-11. The training algorithm has the following features: 
  
* Adam optimizer was used; 
* Learning rate is 0.001; 
* Batch size is 128; 
* Number of epochs is 60;
* Dropout rate is 0.5

I found that with these parameters the training process shows good convergence:

## Solution approach 

As described above, I managed to achieve the accuracy threshold of 0.93 on the validation set by implementing the following
techniques: 

* Images were converted to grayscale; 
* I used the dropout technique to improve performance and prevent overfitting.   

The code for calculating accuracy is located in cells #11, #12, and #15. 

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.942 
* test set accuracy of 0.935

In addition, I explored in more detail the resulting model accuracy on the validation dataset. Code in cell #13 builds and displays 
the confusion matrix, as well as displays the distribution of different image classes in the training and validation dataset. 
It is clear from these visualizations, that the model's performance isn't the same for different classes. They also show that 
the model shows poor performance on classes that are under-represented in the training dataset. It suggests that the accuracy can be further improved if I augment the dataset with more data samples for under-represented classes. 


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 