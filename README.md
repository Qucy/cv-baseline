

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

![alexnet](https://github.com/Qucy/cv-baseline/master/img/alexnet.jpg)

##### 1.3 Training tricks

- Use ReLU as activation function, accelerate training speed, prevent vanished gradient.
- Local Response Normalization (Not used anymore)
- Pooling layers
- Dropout
- Data argumentation, shift, flip and changing colors

##### 1.4 Summary & Conclusion from author

- Their(Model) capacity can be controlled by varying their depth and width. (1 Introduction P2)
- All of our experiments suggest that our results can improved simply by waiting for faster GPU and bigger datasets to become available. (1 Introduction P5)
- ReLU have the desirable property that they do not require input normalization to prevent them from saturating. (3.3 LRN P1)
- The network has learned a variety of frequency- and orientation-selective kernels, as well as various colored blobs. (6.1 P1)
- If two images produce feature activation vectors with a small Euclidean separation, we can say that higher levels of the neural network consider them to be similar. (6.1 P3)
- It is notable that our network's performance degrades if a single convolution layers is removed. (7 Discussion p1)







