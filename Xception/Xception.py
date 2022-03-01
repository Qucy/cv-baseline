import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class XceptionBlock1(layers.Layer):
    """
    Xception block 1 mainly used in Entry flow and Exit flow
    """
    def __init__(self, filter_num):
        super(XceptionBlock1, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (1, 1), strides=(2, 2), padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dwConv1 = layers.SeparableConv2D(filter_num, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        self.bn2 = layers.BatchNormalization()
        self.act = layers.ReLU()
        self.dwConv2 = layers.SeparableConv2D(filter_num, (3, 3), padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.pooling = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')


    def call(self, inputs, *args, **kwargs):
        residual = self.bn1(self.conv1(inputs))
        x = self.dwConv1(inputs)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dwConv2(x)
        x = self.bn3(x)
        x = self.pooling(x)
        x = layers.Add()([x, residual])
        return  x


class XceptionBlock2(layers.Layer):
    """
    Xception block 2 mainly used in Middle flow
    """
    def __init__(self):
        super(XceptionBlock2, self).__init__()

    def call(self, inputs, *args, **kwargs):
        pass


class Xception(Model):
    """
    Xception model
    """
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.entry_flow_b1 = Sequential([
            layers.Conv2D(32, kernel_size=3, strides=2, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.entry_flow_b2 = Sequential


    def call(self, inputs, training=None, mask=None):
        # (b, 299, 299, 3) -> (b, 149, 149, 64)
        x = self.entry_flow_b1(inputs)
        #


        return x


if __name__ == '__main__':
    # test forward pass
    fake_images = tf.random.normal([4, 299, 299, 3])
    xception = Xception()
    outputs = xception(fake_images)
    print(outputs.shape)





