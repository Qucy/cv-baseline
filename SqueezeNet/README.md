### SqueezeNet

##### 1. Abstraction

The paper named <<SqueezeNet: AlexNet-Level accuracy with 50x fewer parameters and <0.5 MB Model Size>>. It mainly raised out 3 points as below:

- Smaller CNNs require less communication across servers during distributed training. 
-  Smaller CNNs require less bandwidth to export a new model from the cloud to an autonomous car
- Smaller CNNs are more feasible to deploy on FPGAs and other hardware with limited memory.

##### 2. SqueezeNet

In SqueezeNet author raised 3 strategies to solve above problems:

- Strategy 1, replace 3x3 filter with 1x1 filter, which can reduce 9x parameters
- Strategy 2, decrease the number of input channels to 3x3 filters
- Strategy 3, downsample late in the network so that convolution layers have large activation maps

Below image depicting the basic module **Fire Module** for SqueezeNet,  a fire module is comprised of: a squeeze convolution layer (which has only 1x1 filters), feeding into an expand layer that has a mix of 1x1 and 3x3 convolution filters; we illustrate this in Figure 1. The liberal use of 1x1 filters in Fire modules is an application of Strategy 1 from Section 3.1. We expose three tunable dimensions (hyperparameters) in a Fire module: s1x1, e1x1, and e3x3. In a Fire module, s1x1 is the number of filters in the squeeze layer (all 1x1), e1x1 is the number of 1x1 filters in the expand layer, and e3x3 is the number of 3x3 filters in the expand layer. When we use Fire modules we set s1x1 to be less than (e1x1 + e3x3), so the squeeze layer helps to limit the number of input channels to the 3x3 filters, as per Strategy 2.

![fire_module](https://github.com/Qucy/cv-baseline/blob/master/img/fire_module.jpg)

Below image illustrate in that SqueezeNet begins with a standalone convolution layer (conv1), followed by 8 Fire modules (fire2-9), ending with a final conv layer (conv10). We gradually increase the number of filters per fire module from the beginning to the end of the network. SqueezeNet performs max-pooling with a stride of 2 after layers conv1, fire4, fire8, and conv10; these relatively late placements of pooling are per Strategy 3.

![squeezeNet](https://github.com/Qucy/cv-baseline/blob/master/img/squeezeNet.jpg)

Below image is the SqueezeNet architecture table.

![squeezeNeta_arc](https://github.com/Qucy/cv-baseline/blob/master/img/squeezeNet_arc.jpg)

##### 3. Model Performance

From table 2 we can see that after apply Deep Compression with 6-bit quantization the smallest model is even less than 0.5 MB which is quite impressive!

![squeezeNet_performance](https://github.com/Qucy/cv-baseline/blob/master/img/squeezeNet_performance.jpg)
