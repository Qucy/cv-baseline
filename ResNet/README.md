### ResNet and ResNeXt

##### 1. Papers

There are two papers regarding ResNet:

- Deep Residual Learning for Image Recognition - CVPR 2016
- Aggregated Residual Transformations for Deep Neural Networks - CVPR 2017

##### 2. ResNet

Before introducing ResNet, we need to know that ResNet is not the first deepest CNN. There is another network called Highway Network, which borrows the ideal from LSTM, add another path with gate in the network forward calculation. It become the first deepest CNN with 900 layers at most. ResNet is kind of borrow the ideal from Highway network and add a short cut in network forward calculation. Due to it's simplicity, ResNet become the most popular network in the industry.

A residual block it's like below, it's adding a short cut X and then concat X with F(X) at the end of block. So the formula can be H(x) = F(x) + x, the reason why this is working well, is because when doing back propagation, the partial derivative of x will always be 1, it's fix the vanished gradient problem for a very deep CNN. That's even a very deep ResNet is still trainable and have a good performance.

![residual_block](https://github.com/Qucy/cv-baseline/blob/master/img/residual_block.jpg)

In the paper author trained several ResNet with different number of layers, the architectural is as below

 ![resnet](https://github.com/Qucy/cv-baseline/blob/master/img/resnet.jpg)

##### 3. ResNeXt

Background for ResNeXt:

- VGG, model architecture is straight forward, use 3x3 conv layers to stack one by one, widely used in all kinds of computer vision tasks
- ResNet, using VGG model architecture and adding residual block
- Inception, using multiple branch or Split-Transform-Merge in the network, but too many parameters and hard to train

ResNeXt borrow all the good ideal from above networks and to produce another powerful model. ResNeXt win the second place in classification task on ILSVRC-2016.

Below is the ResNet block and ResNeXt block, the main difference is inside the block, for ResNet it only have one path and a short cut path. For ResNeXt it borrow the ideal from Inception module it has 32 paths and with a short cut path.

![resNeXt_block](https://github.com/Qucy/cv-baseline/blob/master/img/resNeXt_block.jpg)

Below picture depicting the network architecture for ResNet 50 and ResNeXt 50, the overall architecture is same the only difference is inside the block, ResNeXt is using multiple paths, have less parameter but more conv layers to detect different features.

![ResNet_vs_ResNeXt](https://github.com/Qucy/cv-baseline/blob/master/img/ResNet_vs_ResNeXt.jpg)

Model performance surpass Inception-v4 and Inception-ResNet-v2 on image net 1K datasets.

![ResNeXt_performance](https://github.com/Qucy/cv-baseline/blob/master/img/ResNeXt_performance.jpg)

And on image net 5K datasets(contains 6.8 million images, 5 times larger than 1K datasets), the performance's gab is even bigger.

![ResNeXt_performance_5k](https://github.com/Qucy/cv-baseline/blob/master/img/ResNeXt_performance_5k.jpg)