Arthur: Yehu Chen
Date: 2/12/2018

Reference: https://www.tensorflow.org/tutorials/deep_cnn

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 


The architecture used in this project for CNN of training is:
1. 	conv 3x3, 16
	relu
	max pool, 3x3, stride=2, SAME
	batch normalization
2. 	conv 3x3, 16
	relu
	max pool, 3x3, stride=2, SAME
	batch normalization
3. 	conv 4x4, 16
	relu
	max pool, 3x3, stride=1, SAME
	batch normalization
4.	fully connected, 384
	batch normalization
5.	fully connected, 192
	batch normalization
6.	softmax, 10


In this project, training data is batched into size of 256. When the number of batches 
are chosen to be 10000, the testing accuracy is about 71%. Notice this is a fairly good
result, compared to 86% testing accuracy but the number of batches are 1,000,000.
