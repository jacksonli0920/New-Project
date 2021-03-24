# Image Classification

## Introduction
I am a junior in high school. In this project, I'm using Convolutional Neural Network to achieve Image classification.
I achieved 62% accuracy for the CIFAR-10 class testing data.


## Neutral Network Model arichtechture
Convolutional Neural Networks and CIFAR-10
Neural Network Breakdown![image](https://user-images.githubusercontent.com/74844660/111891932-ad45de80-8a31-11eb-94b7-5f2a79962841.png)
Convolutional neural networks (CNNs) are class of deep learning frameworks primarily designed to work with image data. The core of a CNN is the convolution layer that performs the convolution operation. In mathematics, convolution is an operation that computes the element-wise dot product of an input matrix and a vector. In a CNN, the input matrix is the two-dimensional image while the vector (filter) is the set of weights. Convolutional neural networks gain their power due to translation invariance (Bengio et al. 341). In other words, if a filter is designed to extract or detect a specific feature in an image, the systematic application of the filter across the image will facilitate the discovery of said feature anywhere in the image. This paper is a report on a project to build an image recognition model using the CIFAR-10 dataset. The focus, however, will be on the methods to improve the accuracy of the machine learning model. 


## test data
CIFAR-10 Dataset
The Canadian Institute for Advanced Research (CIFAR-10) is a standard dataset used in computer vision and deep learning to build classification models. The dataset is especially important when learning how to develop, train, evaluate, and implement deep learning neural networks for object recognition. The dataset contains tens of thousands of colored photographs with ten object classes like bird, cat, dog, airplanes, and frog, among other. Each image is unique for every class (Recht et al. 4). When training a convolutional neural network, the recommended range is 50,000 images to be used to train the network and an additional 10,000 images be used as testing images for evaluating the accuracy of the network.

## How to tune the model
Different Testing Functions![image](https://user-images.githubusercontent.com/74844660/111891927-999a7800-8a31-11eb-8d6c-df89ae4efff6.png)
#### learning rate
Smaller the learning rate the more accurate the return result will be. However could cause overfitting to certain models due to too small of a learning rate.
#### loss
It is an error prediction function for the neural network. Calculating the gradients to evaluate the accuracy of the network.
#### batch_size
The number of subjects per iteration, the larger the batch size the subjects tested per iteration.
#### epoch
The number of cycles of tests conducted in one execution of image classification. If the testing set consists of 10,000 subjects, the number of epochs will determine how many cycles of tests will be conducted on the 10,000 subjects.
#### loss curve
A curve graph produced by the loss function of the neural network that demonstrates the progression of loss over trials.

## Conclusion
This paper is a report on a project to build an image recognition model using the CIFAR-10 dataset. A simple convolutional neural network was written in python with two convolutional layers. The model was trained using 10,000 images from the CIFAR-10 dataset with a constant learning rate of 0.001. The resultant network had an accuracy of 62%. What was found to be very effective in decreasing the learning rate and preventing overshooting.  Increasing the epoch allowed the model to have more training samples and allow for better data collection. Different approaches on how to improve the accuracy of the network were discussed. For instance, dropout regularization could be used, where the neural network drops nodes in a random manner during training. The overall outcome is that the network will be less likely to memorize the training dataset and instead learn the generalizable features. Alternatively, data augmentation could be used, where the programmer makes copies some of the elements in the training datasets then applies small random modification. The outcome is that the network both has a larger training dataset while it is being forced to learn the general features of the training set. Regardless of the method used, the accuracy of the resultant network could be improved. 

### Works Cited
Bengio, Yoshua, Ian Goodfellow, and Aaron Courville. Deep learning. Vol. 1. Massachusetts, USA:: MIT press, 2017.
Recht, Benjamin, et al. "Do cifar-10 classifiers generalize to cifar-10?." arXiv preprint arXiv:1806.00451 (2018)

