### VGG Network - Deep Convolution network

##### 1. Abstraction

The paper was published on 2015 by Oxford students Karen and Andrew, the paper named Very Deep Convolution networks for large-scale image recognition.

- The paper researched that the depth of network can increase the model performance
- Use 3*3 kernels to replace 7x7 and 5x5 kernels can improve the network performance
- VGG won second place in classification and first place in object localization
- VGG can generalize to other datasets and perform well
- Open source VGG16 and VGG19

##### 2. Network architecture

Below image depicting the VGG network architecture and different configurations. Column D is VGG16 and column E is VGG19,  16 and 19 means there are 16 or 19 convolutions layers and fully connected layers.

![vgg](https://github.com/Qucy/cv-baseline/blob/master/img/vgg.jpg)

##### 3. VGG features

- Using 3x3 kernels to replace 7x7 and 5x5 kernels(ZFNet), to increase receptive field and reduce parameter. Two 3x3 kernels equals to one 5x5 kernels and three 3x3 kernels equals to one 7x7 kernels.  For example, a 7x7 kernels with C channels the number of parameters equals to 7x7xCxC = 49C^2, three 3x3 kernels with C channel equals to 3x(3x3xCxC) = 27C^2, which reduce almost 44% parameters.
- Try 1x1 kernels (column C)

##### 4. Training tricks

- Data argumentation, scale jittering: scale image to a larger size -> randomly crop image to 224 x 224 -> randomly flip image horizontally
- Data argumentation, modify RGB channel's value to add noise in the color
- Train small network first(column A) and  use small network's weights to initialize deep network (column B,C,D,E)
- Multi-scale training, using different image size to train model and start from small image size (256 ~ 384)

##### 5. Summary and conclusion from author

- use a smaller kernel and deeper network can have a better performance
- use scale jittering in training and testing can improve performance
- LRN does not improve performance
- Xavier initialization have a better performance
- To speed-up training of the S = 384 network, it was initialized with the weights pre-trained with S = 256, and we used a smaller initial learning rate of 0.001.
- Since objects in images can be of different size, multi scale training is beneficial to take this into account during training.