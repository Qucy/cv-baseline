

## CV baseline papers and networks

This is an article help me or you to understand some of baseline papers or networks for computer vision. Before we get to start, we need to the know **ILSVRC**, stands for ImageNet Large Scale Visual Recognition Challenge. It's started by Fei Fei Li since 2010 and held consecutively for 8 years, which push computer vision forward for a lot..a lot...(you know what I mean)

This challenge contains Image Classification, Object localization, Object detection, Object detection from Video, Scene Classification and Scene parsing. Famous models like AlexNet, VGG, GoogleNet, ResNet and DenseNet are come from this competition. We will go through these models one by one.



### 1. AlexNet the first convolution network

##### 1.1 Abstraction

The paper is publish on 2012 by Alex Krizhevsky and named **ImageNet Classification with Deep Convolution Neural Network**. Below is the abstraction for this paper.

- AlexNet was trained on ILSVRC-2010 image dataset which contains 1.2 million images. And win the completion with top-1 and top-5 error are 37.5% and 17%.

- AlexNet is consist by 5 convolution layers and 3 fully connected layers, with over 60 million parameters and 650 K neurons.
- To accelerate training AlexNet using ReLU as activation function and trained on GPU
- Use dropout layers to avoid overfitting
- Surpass second place 10.9% in top-5 error

##### 1.2 Network Architecture

Below image depicting the AlexNet architecture, we will not go through it here, because it quite straight forward here. The only thing I want to mentioned here is the network split into 2 parts because the author trained this 2 parts on different GPU due to memory constrain at that time.

![alexnet](https://github.com/Qucy/cv-baseline/blob/master/img/alexnet.jpg)

##### 1.3 Training tricks

- Use ReLU as activation function, accelerate training speed, prevent vanished gradient.
- Local Response Normalization (Not used anymore)
- Pooling layers
- Dropout
- Data argumentation, shift, flip and changing colors

##### 1.4 Summary & Conclusion from author

- Their(Model) capacity can be controlled by varying their depth and width.
- All of our experiments suggest that our results can improved simply by waiting for faster GPU and bigger datasets to become available.
- ReLU have the desirable property that they do not require input normalization to prevent them from saturating.
- The network has learned a variety of frequency- and orientation-selective kernels, as well as various colored blobs.
- If two images produce feature activation vectors with a small Euclidean separation, we can say that higher levels of the neural network consider them to be similar.
- It is notable that our network's performance degrades if a single convolution layers is removed.


### 2. VGG Network - Deep Convolution network

##### 2.1 Abstraction

The paper was published on 2015 by Oxford students Karen and Andrew, the paper named Very Deep Convolution networks for large-scale image recognition.

- The paper researched that the depth of network can increase the model performance
- Use 3*3 kernels to replace 7x7 and 5x5 kernels can improve the network performance
- VGG won second place in classification and first place in object localization
- VGG can generalize to other datasets and perform well
- Open source VGG16 and VGG19

##### 2.2 Network architecture

Below image depicting the VGG network architecture and different configurations. Column D is VGG16 and column E is VGG19,  16 and 19 means there are 16 or 19 convolutions layers and fully connected layers.

![vgg](https://github.com/Qucy/cv-baseline/blob/master/img/vgg.jpg)

##### 2.3 VGG features

- Using 3x3 kernels to replace 7x7 and 5x5 kernels(ZFNet), to increase receptive field and reduce parameter. Two 3x3 kernels equals to one 5x5 kernels and three 3x3 kernels equals to one 7x7 kernels.  For example, a 7x7 kernels with C channels the number of parameters equals to 7x7xCxC = 49C^2, three 3x3 kernels with C channel equals to 3x(3x3xCxC) = 27C^2, which reduce almost 44% parameters.
- Try 1x1 kernels (column C)

##### 2.4 Training tricks

- Data argumentation, scale jittering: scale image to a larger size -> randomly crop image to 224 x 224 -> randomly flip image horizontally
- Data argumentation, modify RGB channel's value to add noise in the color
- Train small network first(column A) and  use small network's weights to initialize deep network (column B,C,D,E)
- Multi-scale training, using different image size to train model and start from small image size (256 ~ 384)

##### 2.5 Summary and conclusion from author

- use a smaller kernel and deeper network can have a better performance
- use scale jittering in training and testing can improve performance
- LRN does not improve performance
- Xavier initialization have a better performance
- To speed-up training of the S = 384 network, it was initialized with the weights pre-trained with S = 256, and we used a smaller initial learning rate of 0.001.
- Since objects in images can be of different size, multi scale training is beneficial to take this into account during training.


### 3. Google Net - Going deeper and wider

##### 2.1 Abstraction

Below are 4 papers published for Google Net from 2015 to 2017, we're going to look at key points in these papers.

- Going Deeper with convolutions - CVPR 2015
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift - 2015
- Rethinking the Inception Architecture for Computer Vision - 2015
- Inception V4, Inception ResNet and the Impact of Residual Connection on Learning - AAAI 2017

The performance for all the Google Net is as below.

![gooleNetPerformance](https://github.com/Qucy/cv-baseline/blob/master/img/gooleNetPerformance.JPG)

##### 2.2 Network Architecture

The overall architecture for GoogleNet V1 is as below, it consists by lots of Inception module(highlighted in green box) and have 2 middle layer output highlighted in blue box. The new version has the similar overall architecture, so we will not demonstrate here. 

![googleNet](https://github.com/Qucy/cv-baseline/blob/master/img/googleNet.jpg)



##### 2.3 GoogleNet V1

- ILSVRC 2014 classification first place, object detection first place, object localization second place

- Introduce Inception module, use multiple dimensions filters(1x1, 3x3, 5x5) (borrow the ideal from Gabor filters), use 1x1 convolution to reduce dimensions.

  ![inceptionV1](https://github.com/Qucy/cv-baseline/blob/master/img/inceptionV1.jpg)

- Adding middle layer outs for loss calculation to prevent vanished gradient problem

- Use learning rate decay policy, decreasing learning rate 4% every 8 epochs

- Photometric distortions in data argumentation is useful(adjust Brightness, Saturation and Contrast)

- Use multi-crop in testing phase, crop 1 image to 144 images.

- Use ensembled models(7 models) to produce ensembled predictions.

##### 2.4 GoogleNet V2

