import os
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class InvertResidualBlock(layers.Layer):

    def __init__(self, filters, strides=1, padding='valid'):
        """
        initialize method for InvertResidualBlock
        :param filters: number of filters
        :param strides: number of strides for DW Conv
        :param padding: padding for DW Conv
        """
        super(InvertResidualBlock, self).__init__()
        self.pwConv1 = layers.Conv2D(filters, kernel_size=1, strides=1)
        self.bn1 = layers.BatchNormalization()
        self.dwConv = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding=padding)
        self.bn2 = layers.BatchNormalization()
        self.pwConv2 = layers.Conv2D(filters, kernel_size=1, strides=1)
        self.bn3 = layers.BatchNormalization()

    def call(self, inputs, *args, **kwargs):

        x = self.dwConv(inputs)
        x = self.bn1(x)
        x = tf.nn.relu6(x)
        x = self.pwConv(x)
        x = self.bn2(x)
        x = tf.nn.relu6(x)

        return x


class MobileNetV2(Model):

    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.input_block = Sequential([
            layers.Conv2D(32, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.pooling = layers.AveragePooling2D(7)
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        # b,224,224,3 -> b,112,112,32
        x = self.input_block(inputs)

        # b,7,7,1024 -> b,1,1,1024
        #x = self.pooling(x)
        # b,1,1,1024 -> b,1,1,1000
        #x = self.fc(x)

        return x


if __name__ == '__main__':
    # test forward pass
    fake_images = tf.random.normal([4, 224, 224, 3])
    mobileNetV2 = MobileNetV2()
    outputs = mobileNetV2(fake_images)
    print(outputs.shape)