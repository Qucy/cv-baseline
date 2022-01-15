### SENet

##### 1. Abstraction

SENet won 1st place of ILSVRC 2017 image classification challenge, the paper named Squeeze-and-Excitation Networks and published on CVPR 2018. Because SENet achieve 2.25% top 5 error and people think due to noise in the dataset, it is better not to improve accuracy further to avoid overfitting, hence 2017 is the last competition for ILSVRC. SENet using the attention mechanism in computer vision and before that attention is widely used in NLP.

##### 2. SE block

Below image depicting the SE block:

- apply global average pooling to the inputs, so feature shape will change from H,W,C to 1,1,C
- apply 2 fully connected layer, the first fc layer with C/R neurons and 2nd fc layer with C neurons which is same as input
- apply a sigmoid function to the second fc layers' output, to retrieve the channels weights
- multiple channels weights with original feature maps

![se_block](https://github.com/Qucy/cv-baseline/blob/master/img/se_block.jpg)

Here we call it SE block, because in this paper the author didn't invent any new network. SE block you can think it's like a Lego block can append into any network. Below image demonstrate how SE block can be appended into a ResNet.

![se_block_resnet](https://github.com/Qucy/cv-baseline/blob/master/img/se_block_resnet.jpg)

Below table describe the network architecture for ResNet-50, SE-ResNet-50 and SE-ResNeXt-50, there is no much bigger difference, the only difference for SE version is that SE block is appended at the end of each residual block.

![se_ResNeXt](https://github.com/Qucy/cv-baseline/blob/master/img/se_ResNeXt.jpg)

##### 3. Model performance

The author tried a lot experiments in this paper, below image depicting the performance before and after SE block is appended in the network. We can see every network have difference performance. And for SE-ResNet-50 has similar performance with ResNet-101 but only have 50% of it's GFLOPs.

![se_p1](https://github.com/Qucy/cv-baseline/blob/master/img/se_p1.jpg)

The author also append SE block into light weighted models like Mobile Net and Shuffle Net. And the performance also improved as below.

![se_p2](https://github.com/Qucy/cv-baseline/blob/master/img/se_p2.jpg)

The author also tried with different dataset like CIFAR-10, CIFAR-100, Place365 and COCO. The performance also improved after integrated with SE block.

![se_p3](https://github.com/Qucy/cv-baseline/blob/master/img/se_p3.jpg)

All of the above experiments shows that SE block is very useful block not limit to task and datasets.

##### 4. Other attention

Beside SE attention there are other 2 attention are been used a lot in computer vision. The first one is **CBAM attention.** Below image depicting the CBAM attention. It can be divide into 2 parts the first parts is Channel Attention which is quite similar with SE attention.

**Channel attention**

- calculate max pooling and average pooling use input feature maps
- send max pooling results and average pooling results into a shared fc layers
- add 2 results and apply sigmoid function to get channel attention
- multiply channel attention with original feature maps

**Spatial Attention**

- calculate max and average value for each feature point
- concat 2 results and apply a 1 by 1 conv layers to adjust channels
- apply a sigmoid function to get spatial attention
- multiply spatial attention with original feature maps

![CBAM](https://github.com/Qucy/cv-baseline/blob/master/img/CBAM.jpg)

Another attention is used in ECANet, we can call it **Efficient Channel Attention**. It's a upgrade version compared with SE block. The author think that SE block capture all the channel attention is low efficient and will have side affects. So the author proposed that use 1 by 1 conv layers to replace fc layers in the Excitation part, because author thought that conv layers is capable to capture information from multiple channels.

![eca_block](https://github.com/Qucy/cv-baseline/blob/master/img/eca_block.jpg)