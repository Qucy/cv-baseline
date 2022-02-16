### MobileNet

##### 1. Abstraction

There are 3 versions MobileNet, the first version is MobileNet V1, the key point is that to use depthwise separable convolution to reduce parameters and calculation. After V1, V2 and V3 combine the advantage from other networks like residual and attention. The research background for MobileNet is just as its names, want to develop a lightweight model without losing too many accuracy that can be used in Mobile devices and embedded devices. So let's take a look at these networks.

##### 2. MobileNet V1

The main trick in MobileNet V1 is to use depthwise separable convolution to replace normal convolutional layers. But before we dive into the details let's take a look at the overall architecture of MobileNet V1. Below image depicting the architecture of MobileNet V1, the Conv stands for the normal convolutional layers while **Conv dw** stands for depthwise convolutional layers. And after each **Conv dw** layers there is a point wise convolutional layers. 

![mobileNetV1](https://github.com/Qucy/cv-baseline/blob/master/img/mobileNetV1.jpg)

Below image compare the normal convolutional block(left) and depthwise convolutional block(right). A normal conv block contains a conv layer, a BN layer and a activation layer. For DW layer it contains a DW Conv layer, a BN a layer, a activation layer, a point wise conv layer, a BN layer and activation layer again.

![dw_module](https://github.com/Qucy/cv-baseline/blob/master/img/dw_module.jpg)

Below image demonstrate the detailed difference between normal conv layers and depthwise conv layers.

- A normal conv layers' filters will calculate feature map across all the channels and with a specified strides. That's why the spatial connection between input and output is sparse and channel connection is dense.
- A depthwise conv layers filters will calculate feature maps separately for each channels, so the connection for spatial is same as normal conv but the connection for channel is separate.
- A pointwise conv is just a normal conv but with a 1x1 filter and strides equals to 1, so the connection for spatial is separate and for channel is a dense connection. The reason the author add a pointwise conv layers is to enable information exchange between different channels.

![normal_conv_vs_dw_conv](https://github.com/Qucy/cv-baseline/blob/master/img/normal_conv_vs_dw_conv.jpg)

Below is the summary table between normal conv layers and DW conv layers.

![normal_conv_vs_dw_conv_detail](https://github.com/Qucy/cv-baseline/blob/master/img/normal_conv_vs_dw_conv_detail.jpg)



##### 3. MobileNet V2 and MobileNet V3

**MobileNet V2**, inverted residual blocks and Linear bottlenecks. (2018)

- Linear bottlenecks, the RELU function will break the features in a low dimensional space. So in MobileNet V2 the author removed RELU layers after reduce the dimension in a conv layer.

![relu_break](https://github.com/Qucy/cv-baseline/blob/master/img/relu_break.jpg)

- Inverted residual, in the original residual blocks the author use a 1x1 conv to reduce dimension and then use another 3x3, 1x1 conv layers to up sampling the data. But for MobileNet since the dimension is smaller than original ResNet, in order to avoid losing features, so the author propose to increase the dimension first and then reduce dimension again. That's it called inverted residual blocks.

  ![inverted_residual](https://github.com/Qucy/cv-baseline/blob/master/img/inverted_residual.jpg)

  Below is the architecture for MobileNet V2, totally 54 layers.

![MobileNetV2](https://github.com/Qucy/cv-baseline/blob/master/img/MobileNetV2.jpg)



**MobileNet V3**,  Searching for MobileNet V3 (2019)

- new activation function h-swish, h-swish is a improved version compared with swish and in h-swish author use RELU to replace sigmoid function in order to achieve a better performance on mobile device.

  ![h-swish](https://github.com/Qucy/cv-baseline/blob/master/img/h-swish.jpg)

- SENet, add attention SENet into MobileNet V3. For SENet please refer to (https://github.com/Qucy/cv-baseline/tree/master/SENet)



##### 4. Model Performance

Below are model performance for MobileNet V3 in difference tasks.

![MobileNet_performance](https://github.com/Qucy/cv-baseline/blob/master/img/MobileNet_performance.jpg)
