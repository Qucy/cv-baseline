### AlexNet the first convolution network

##### 1. Abstraction

The paper is publish on 2012 by Alex Krizhevsky and named **ImageNet Classification with Deep Convolution Neural Network**. Below is the abstraction for this paper.

- AlexNet was trained on ILSVRC-2010 image dataset which contains 1.2 million images. And win the completion with top-1 and top-5 error are 37.5% and 17%.

- AlexNet is consist by 5 convolution layers and 3 fully connected layers, with over 60 million parameters and 650 K neurons.
- To accelerate training AlexNet using ReLU as activation function and trained on GPU
- Use dropout layers to avoid overfitting
- Surpass second place 10.9% in top-5 error

##### 2. Network Architecture

Below image depicting the AlexNet architecture, we will not go through it here, because it quite straight forward here. The only thing I want to mentioned here is the network split into 2 parts because the author trained this 2 parts on different GPU due to memory constrain at that time.

![alexnet](https://github.com/Qucy/cv-baseline/blob/master/img/alexnet.jpg)

##### 3. Training tricks

- Use ReLU as activation function, accelerate training speed, prevent vanished gradient.
- Local Response Normalization (Not used anymore)
- Pooling layers
- Dropout
- Data argumentation, shift, flip and changing colors

##### 4. Summary & Conclusion from author

- Their(Model) capacity can be controlled by varying their depth and width.
- All of our experiments suggest that our results can improved simply by waiting for faster GPU and bigger datasets to become available.
- ReLU have the desirable property that they do not require input normalization to prevent them from saturating.
- The network has learned a variety of frequency- and orientation-selective kernels, as well as various colored blobs.
- If two images produce feature activation vectors with a small Euclidean separation, we can say that higher levels of the neural network consider them to be similar.
- It is notable that our network's performance degrades if a single convolution layers is removed.

