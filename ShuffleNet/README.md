### ShuffleNet

##### 1. Abstraction

There are 2 versions for ShuffleNet and the key point for ShuffleNet is to use group convolution and channel shuffle to reduce calculation and keep a good accuracy.

##### 2. ShuffleNet V1

To understand ShuffleNet V1 is mainly to understand what is **group convolution**. Below image depicting the normal convolution(left) vs the group convolution(right).  

- input channels = M, output channel = N, kernel size = 3x3, group = 3
- normal conv calculation = 3 x 3 x M x N
- group conv calculation = 3 x 3 x (M/3) x (Nx3) x 3

In this example you can save 3 times calculation compared with normal convolution. And this number depends on how many groups you are using.

![grp_conv](https://github.com/Qucy/cv-baseline/blob/master/img/grp_conv.jpg)

And one more thing need to understand is **channel shuffle**. Because author use group convolution here, it will constraint the information communication between different channel groups. Hence author propose to use channel shuffle to handle this problem. Below image depicting how channel shuffle works. Suppose we have 3 groups here and each group can be divided into 4 sub groups. First reshape it to (3, 4) , transpose it to (4, 3) and at last flatten it.

![channel_shuffle](https://github.com/Qucy/cv-baseline/blob/master/img/channel_shuffle.jpg)

The overall architecture for ShuffleNet V1 is as below.

![shuffleNetV1](https://github.com/Qucy/cv-baseline/blob/master/img/shuffleNetV1.jpg)

##### 3. ShuffleNet V2

For ShuffleNet V2 author raise out 4 points:

- When number of input channel and number of output channel is same, have the best performance in terms of speed
- Too much group convolution increase network latency
- Too much shortcut increase fragment and network latency
- Element wise operation like RELU and ADD increase network latency

Based on these 4 points, author introduced ShuffleNet V2,

- use channel split before shortcut
- replace group conv by standard point wise conv (prevent too much group conv)
- put channel shuffle after concate with short cut (prevent too much fragment)
- replace add operation with concate

![shufflev1_vs_v2](https://github.com/Qucy/cv-baseline/blob/master/img/shufflev1_vs_v2.jpg)

Below image is the overall architecture for ShuffleNet V2

![shuffleNetV2](https://github.com/Qucy/cv-baseline/blob/master/img/shuffleNetV2.jpg)

##### 4. Model Performance

![shuffleNetV2_performance](https://github.com/Qucy/cv-baseline/blob/master/img/shuffleNetV2_performance.jpg)
