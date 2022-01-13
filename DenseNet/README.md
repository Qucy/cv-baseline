### DenseNet - Densely Connected Convolutional Networks

##### 1. Abstraction

Paper name is just as this title, Densely Connected Convolutional Networks and published on CVPR 2017 by Gao Huang & Zhuang Liu. The main change in this network is Dense Block. You think it's an upgrade residual block.

##### 2. Dense block

A dense block is looks like below, in the dense block every layer will receive all the feature maps from previous layers and all the outputs will pass to the layers later. So in dense block with N layers, the total number of connections will be N * (N + 1) / 2. The advantage by using dense block can be summarized as below:

- reduce the parameters in the network while having more feature maps
- reuse lower level feature maps, make feature maps more meaningful at end
- stronger gradient flow in back propagation

Dense block
![denseBlock](https://github.com/Qucy/cv-baseline/blob/master/img/denseBlock.jpg)

High level architecture for dense net
![denseNet](https://github.com/Qucy/cv-baseline/blob/master/img/denseNet.jpg)

Dense net with 121, 169,  201 and 264 layers network structure
![denseNet_table](https://github.com/Qucy/cv-baseline/blob/master/img/denseNet_table.jpg)

For more you can read this article which I found very useful for understanding DenseNet architecture -> https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a

##### 3. Model performance

DenseNet Vs ResNet in ILSRVC classification problem
![dense_p1](https://github.com/Qucy/cv-baseline/blob/master/img/dense_p1.jpg)

DenseNet Vs ResNet on CIFAR-10 dataset
![dense_p2](https://github.com/Qucy/cv-baseline/blob/master/img/dense_p2.jpg)