### Google Net - Going deeper and wider

##### 1. Abstraction

Below are 4 papers published for Google Net from 2015 to 2017, we're going to look at key points in these papers.

- Going Deeper with convolutions - CVPR 2015
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift - 2015
- Rethinking the Inception Architecture for Computer Vision - 2015
- Inception V4, Inception ResNet and the Impact of Residual Connection on Learning - AAAI 2017

The performance for all the Google Net is as below.

![gooleNetPerformance](https://github.com/Qucy/cv-baseline/blob/master/img/gooleNetPerformance.JPG)

##### 2. Network Architecture

The overall architecture for GoogleNet V1 is as below, it consists by lots of Inception module(highlighted in green box) and have 2 middle layer output highlighted in blue box. The new version has the similar overall architecture, so we will not demonstrate here.

![googleNet](https://github.com/Qucy/cv-baseline/blob/master/img/googleNet.jpg)



##### 3. GoogleNet V1

- ILSVRC 2014 classification first place, object detection first place, object localization second place

- Introduce Inception module, use multiple dimensions filters(1x1, 3x3, 5x5) (borrow the ideal from Gabor filters), use 1x1 convolution to reduce dimensions.

  ![inceptionV1](https://github.com/Qucy/cv-baseline/blob/master/img/inceptionV1.jpg)

- Adding middle layer outs for loss calculation to prevent vanished gradient problem

- Use learning rate decay policy, decreasing learning rate 4% every 8 epochs

- Photometric distortions in data argumentation is useful(adjust Brightness, Saturation and Contrast)

- Use multi-crop in testing phase, crop 1 image to 144 images.

- Use ensembled models(7 models) to produce ensembled predictions.

##### 4. GoogleNet V2

- Raised Batch Normalization to reduce ICP(Internal Covariate Shift), by using batch normalization we can:
- ​    A bigger learning rate to accelerate model converge
- ​    Don't need to carefully initialize model weights
- ​    Don't need to use drop out layer or drop out layer with small drop out rate
- ​    Don't need to use L2 normalization for model weights
- ​    Don't need to use LNR (local response normalization)
- High level model Architecture change compared with GoogleNet V1:
- ​    Add BN layer before activation
- ​    Borrow ideal from VGG use two 3x3 kernels to replace one 5x5 kernels in Inception module
- ​    Adding more inception layers and use stride of 2

##### 5. GoogleNet V3

Research background for this paper:

- GoogleNet V1 raised multiple dimension kernels, 1x1 kernels and auxiliary classifier to help calculate loss prevent vanished gradient problem
- GoogleNetV2 raised Batch Normalization to accelerate model converge
- VGG network has too much parameter and operations which is not friendly to industry

Base on these, author raised inception V2 and inception V3 module in this paper. Inception V3 become the most common and popular inception module.

We know that a 5x5 convolutional layer can be replaced by two 3x3 convolutional layers. And in this paper author raised asymmetric convolutions, a 3x3 convolution layer can be factorized into a 1x3 and a 3x1 convolution layer. Below image depicting how inception module upgrade, from left to right:

- First image shows that a 3x3 can be replaced by 1x3 and 3x1 convolutional layers
- Second image shows the original inception module in GoogleNet V1
- Third image shows 5x5 convolution layers factorized into two 3x3 convolutional layers in GoogleNet V2
- Forth image shows, a n x n conv layer can factorized into a 1xn and a nx1 asymmetric convolutions
- Fifth image shows, 3x3 convolution layers factorized into two asymmetric convolutions layers a 1x3 and a 3x1 convolutional layers, for a 3x3 layer the parameter will reduce around 33%(1 - (3+3)/9 ) percentage.

![inception_v3](https://github.com/Qucy/cv-baseline/blob/master/img/inception_v3.jpg)

**Efficient grid size reduction** is another ideal raised in this paper, below image depicting how it works

- The first image shows the left one violates the size reduction principle because it reduce 35x35 to 17x17 while still keep the channels same as 320, this will loss information. So a more proper way is, first double channels size to 640 and then reduce feature size from 35x35 to 17x17. But right one is 3 times more expensive computationally.
- So the author proposed instead of double channels directly, we use a 17x17 conv and a 17x17 pool layer and then concat both layers.(The 2nd image)

![effcient_grid_size](https://github.com/Qucy/cv-baseline/blob/master/img/effcient_grid_size.jpg)

**Label smoothing** is a another trick used during training, the basically idea is that to prevent overfitting, we don't want to model to predict 0 or 1, instead want model to approximate to 0 or 1like, 0.0005 or 0.9995.

Let's take a look how **inception V2 model architecture** changed compared with Inception V1:

- use three 3x3 conv layers to replace 7x7 conv layers at beginning and using strides of 2 to reduce input size
- use another three 3x3 conv layers and use strides of 2 in second 3x3 conv layers
- Add 3 Inception module with 35x35 input size, only replace one 5x5 conv by two 3x3 in the inception module
- Inception module with input size 17x17, use asymmetric conv layers
- Inception module with input size 8x8, follow principle 2 use inception module in figure 7
- linear layer input size is 2048, in V1 is 1024

![inceptionV2](https://github.com/Qucy/cv-baseline/blob/master/img/inceptionV2.jpg)

**Inception V3** has 4 changes compared with Inception V2:

- use RMSProp as optimization function
- use label smoothing
- use asymmetric conv layers to extract 17x17 features
- use BN layers in auxiliary classifier

**Inception V3 performance**, Inception V3 use 4 ensemble inception models to achieve 3.58% top 5 error and become SOTA at that time.

![inception_v3_performance](https://github.com/Qucy/cv-baseline/blob/master/img/inception_v3_performance.jpg)

##### 6. GoogleNet V4

**Research background**, to combine the ResNet into Inception model, to achieve a construct a better model. In the paper the author construct 3 models

- Inception V4, consist of stem(9 layers), Inception A(3x4=12 layers), B(5x7=35 layers), C(3x4=12 layers) and Reduction A(3 layers), B(3 layers), Pooling (1 layer)) in total 76 layers
- Inception ResNet V1, consist of Stem(7 layers), Inception-ResNet A(5x4=20 layers), B(3 layers), C(10x4=40 layers) and Reduction A(3 layers), B(5x4=20 layers), Pooling (1 layer) = 94 layers
- Inception ResNet V2, consist of Stem(9 layers), Inception-ResNet A(5x4=20 layers), B(3 layers), C(10x4=40 layers) and Reduction A(3 layers), B(5x4=20 layers), Pooling (1 layer) = 96 layers

Below image depicting the overall architecture for Inception V4(left) and Inception ResNet V1 and V2 (right)

 ![InceptionV4_InceptionResNet](https://github.com/Qucy/cv-baseline/blob/master/img/InceptionV4_InceptionResNet.jpg)

**Model performance**

- Inception-ResNet-V2 > Inception V4 > Inception-ResNet-V1
- 144 crop better than 12 crop
- Ensemble model use one Inception V4 and 3 Inception-ResNet V2
- Top 5 error 3.1% better than ResNet 152 3.6%

![GoogleNetV4_result](https://github.com/Qucy/cv-baseline/blob/master/img/GoogleNetV4_result.jpg)