import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'


class FireModule(layers.Layer):
    """
    Basic module for SqueezeNet
    """
    
    def __init__(self, squeeze_filters, expand_filters):
        super(FireModule, self).__init__()
        self.s1 = layers.Conv2D(squeeze_filters, kernel_size=1, strides=1, activation=tf.nn.relu)
        self.e1 = layers.Conv2D(expand_filters, kernel_size=1, strides=1)
        self.e3 = layers.Conv2D(expand_filters, kernel_size=3, strides=1, padding='same')
        self.act = layers.ReLU()


    def call(self, inputs, *args, **kwargs):
        # squeeze
        s1 = self.s1(inputs)
        # expand
        x = layers.concatenate([self.e1(s1), self.e3(s1)], axis=-1)
        x = self.act(x)
        return x


class SqueezeNet(Model):
    """
    SqueezeNet model: paper https://arxiv.org/abs/1602.07360
    """
    
    def __init__(self, num_classes):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.input_blocks = Sequential([
            layers.Conv2D(filters=96, kernel_size=7, strides=2, padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=(3,3), strides=2)
        ])
        self.f2 = FireModule(16, 64)
        self.f3 = FireModule(16, 64)
        self.f4 = FireModule(32, 128)
        self.p4 = layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.f5 = FireModule(32, 128)
        self.f6 = FireModule(48, 192)
        self.f7 = FireModule(48, 192)
        self.f8 = FireModule(64, 256)
        self.p8 = layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.f9 = FireModule(64, 256)
        self.c10 = layers.Conv2D(filters=num_classes, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
        self.p10 = layers.GlobalAveragePooling2D()


    def call(self, inputs, training=None, mask=None):
        # (b, 224, 224, 3) -> (b, 112, 112, 96) -> (b, 55, 55, 96)
        x = self.input_blocks(inputs)
        # (b, 55, 55, 96) -> (b, 55, 55, 128)
        x = self.f2(x)
        # (b, 55, 55, 128) -> (b, 55, 55, 128)
        x = self.f3(x)
        # (b, 55, 55, 128) -> (b, 55, 55, 256)
        x = self.f4(x)
        # (b, 55, 55, 256) -> (b, 27, 27, 256)
        x = self.p4(x)
        # (b, 27, 27, 256) -> (b, 27, 27, 256)
        x = self.f5(x)
        # (b, 27, 27, 256) -> (b, 27, 27, 384)
        x = self.f6(x)
        # (b, 27, 27, 384) -> (b, 27, 27, 384)
        x = self.f7(x)
        # (b, 27, 27, 384) -> (b, 27, 27, 512)
        x = self.f8(x)
        # (b, 27, 27, 512) -> (b, 13, 13, 512)
        x = self.p8(x)
        # (b, 13, 13, 512) -> (b, 13, 13, 512)
        x = self.f9(x)
        # (b, 13, 13, 512) -> (b, 13, 13, 1000)
        x = self.c10(x)
        # (b, 13, 13, 1000) -> (b, 1000)
        x = self.p10(x)
        return x



if __name__ == '__main__':
    # test forward pass
    fake_images = tf.random.normal([4, 224, 224, 3])
    squeezeNet = SqueezeNet(num_classes=1000)
    outputs = squeezeNet(fake_images)

    print(outputs.shape)