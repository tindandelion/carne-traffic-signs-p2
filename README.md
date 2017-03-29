# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[unreg-training-progress]: ./output/unregulated-loss.png 
[dropout-training-progress]: ./output/dropout-loss.png 
[real-traffic-signs]: ./output/real-signs-color.png
[real-traffic-signs-classified]: ./output/real-signs-classified.png
[layer-1-vis]: ./output/layer-1-vis.png

## Submitted files

As required by project's rubric, the following files are submitted for the
review: 

1. [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb) The Jupyter notebook containing the project's
code;
2. [report.html](./report.html) The notebook exported into HTML file;
3. [README.md](./README.md) The project write-up (this file)


## Data Set Summary & Exploration

The code for this step is contained in the cell #4 of the notebook. As
long as the dataset is loaded in from of numpy array, most summary points are
immediately available:

* The size of training set is 34799 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32x32 px with 3 color channels (RGB)
* The number of unique classes/labels in the data set is 43 (0-42)

The code in cell #5 visualizes a sample content of the training dataset. To
better understand the challenge, I plot 10 randomly selected images for each
label. This visualization shows how diverse the images belonging to the same
class are.

Later in the project I also explore how many samples of each class there are in
training and validation dataset (see below in "Solution approach" section). It
turns out that the training dataset is not evenly distributed among different
image classes. This fact should be taken into account when I analyze the model's
performance.

## Data preprocessing 

As a preprocessing step, I convert images to grayscale (cell #7). It was
mentioned in
[this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that
grayscale images may show better performance. Indeed, my experiments in this
project showed the prediction accuracy gain from ~0.87 to ~0.9 when I
implemented this step.
   
I decided not to do any further preprocessing, because I managed to achieve
required accuracy threshold of 0.93 with other techniques, described below.

## Model architecture

As a starting point, I used the LeNet-5 network architecture. The network
without modifications showed prediction accuracy of 0.87 on the validation
dataset, which I managed to raise to 0.9 after converting the images to
grayscale.

When training the network, I noticed that the training error was decreasing to
almost 0, but the validation error started to increase after a certain
iteration, which signified that the model was overfitting: 

![unreg-training-progress]

To prevent overfitting and improve performance, I decided to try using dropout
in fully connected layers. Having experimented with different dropout rates, I
found that the dropout rates of 0.3 through 0.6 were giving the best results on
the validation dataset. I decided to use the dropout rate of 0.5 eventually. 

## Final network architecture 

The code that constructs the final network is located in the cell #8. 
My final model consists of the following layers:

| Layer function | Inputs | Outputs |
| -------------- | ------ | ------- |
| Convolutions 5x5, VALID | 32x32x1 | 28x28x6 |
| ReLU | | |
| Max pooling 2x2 | 28x28x6 | 14x14x6 | 
| Convolutions 5x5, VALID | 14x14x6 | 10x10x16 |
| ReLU | | |
| Max pooling 2x2 | 10x10x16 | 5x5x16 |
| Flatten | 5x5x16 | 400 |
| Dropout | | |
| Fully connected | 400 | 120 |
| ReLU | | |
| Fully connected | 120 | 84 |
| ReLU | | | 
| Dropout | | |
| Fully connected | 84 | 43 |

The final fully connected layer produces unscaled logits at output.

## Model training 

The code for training the model is located in cells #9-11. The training algorithm has the following features: 
  
* Optimizer: Adam;
* Learning rate is 0.001; 
* Batch size is 128; 
* Number of epochs is 60;
* Dropout rate is 0.5

I found that with these parameters the training process shows good convergence:

![dropout-training-progress]

## Solution approach 

As described above, I managed to achieve the accuracy threshold of 0.93 on the
validation set by implementing the following techniques:

* Images were converted to grayscale; 
* I used the dropout technique to improve performance and prevent overfitting.   

The code for calculating accuracy is located in cells #11, #12, and #15. 

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.942 
* test set accuracy of 0.935

In addition, I explored in more detail the resulting model accuracy on the
validation dataset. Code in cell #13 builds and displays the confusion matrix,
as well as displays the distribution of different image classes in the training
and validation dataset.  

One insight from these visualizations is that the model's performance varies for
different classes. They also indicate that the model shows relatively poor
performance on classes that are under-represented in the training dataset. It
suggests that the accuracy can be further improved if I augment the dataset with
more data samples for under-represented classes.

## Test the model on new images

The task in the project was to search for traffic sign images on the Internet,
but I live in Finland, so I decided to go out and take pictures of different
traffic signs in my neighbourhood. Finnish traffic signs are similar to German,
but sometimes use different colors and slightly different shapes. 

Here are the pictures I prepared for testing:

![real-traffic-signs]

Notice that 2 images: `10.jpg` and `9.jpg`, were not present in the training
dataset at all. Images `12.jpg` and `7.jpg` are taken from the side, which can
make them harder to recognize. Image `3.jpg` has a square plate, unlike those in
the dataset.

The network classified these images as follows:

![real-traffic-signs-classified]

The code that calculates predictions is located in the cell #17 in the notebook. 
The model was able to correctly identify 100% of the signs it was trained to
classify. As for new sign classes, the prediction actually was quite close. 

Compared to the accuracy on the provided test dataset (0.935), the accuracy on
the new images (1.0) suggests that the model generalizes well over the new
images, i. e. there's no indication of overfitting.

## Softmax probabilities

The code to calculate the softmax probabilities for the new images is provided
in the cell #19. Remarkably, for all known image classes, the model is quite
confident in its predictions: all classes are identified with the probability
of 1.0 or close. 

Even for the unknown classes the top predicted values are around 0.9. 

## Visualize the network activations

As a final step, I visualize the activations from the first convolutional layer,
before the max pooling. The end result is presented in the cell #21 of the
notebook. This visualization shows that the network learned to recognize the
overall shape of the traffic sign, as well as the details of the drawing inside
the frame. 

![layer-1-vis]
