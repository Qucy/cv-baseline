### Xception

##### 1. Abstraction

The paper named <<Xception: Deep Learning with Depthwise Separable Convolutions>>.  The paper present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.

##### 2. Xception

Below image depicting the Xception architecture, it can be divide into 3 parts, entry flow, middle flow and exit flow.

![xception](https://github.com/Qucy/cv-baseline/blob/master/img/xception.jpg)

##### 3. Model Performance

![xception_performance](https://github.com/Qucy/cv-baseline/blob/master/img/xception_performance.jpg)
