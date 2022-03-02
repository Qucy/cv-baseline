### Knowledge distilling

##### 1. Abstraction

The paper is called Distilling the Knowledge in a Neural Network, published by Google on NIPS 2014.  In this paper author introduced a way use a teacher network to teach a small network, so that the small network can have the similiar performance compared with teacher network, and author call it knowledge distilling.

##### 2. Knowledge distilling

Below image depicting the overall architecture of knowledge distilling. On the left side is big model or teacher model, on the right side is the small model or student model.

- the first step is that to train one or more teacher models to make it converge to good accuracy or small error
- the second step is to build a small student network and train it with teacher models together
- total loss = soft loss + hard loss
- soft loss = cross entropy loss between teachers' and students' Softmax function (both need to be divided by T)
- hard loss = cross entropy loss between student prediction and target labels

![knowledge_distilling](https://github.com/Qucy/cv-baseline/blob/master/img/knowledge_distilling.jpg)

And the T in the above image means the temperature, when the T keeps increasing the probability will become smoother.

![kd_temperture](https://github.com/Qucy/cv-baseline/blob/master/img/kd_temperture.jpg)

The reason to introduce this T parameter is that the author hope the student model can learn the logits distribution from teacher network.  Below images shows that after temperature the smallest value can provide more information compared with before.

![kd_softmax_temperture](https://github.com/Qucy/cv-baseline/blob/master/img/kd_softmax_temperture.jpg)

##### 3. Research result

- MINIST dataset - teacher network with 2 hidden layers and 1200 units per hidden layer has 47 errors on test dataset, another network with 2 hidden layers and 800 per hidden units has146 test errors.  After using knowledge distilling the small network with 2 hidden layers and 300 unit per layer have around 74 test errors.

- Speech recogintion - after using the knowledge distilling the model has the similar performance compared with 10x Ensemble models

  ![kd_speech_reco](https://github.com/Qucy/cv-baseline/blob/master/img/kd_speech_reco.jpg)
