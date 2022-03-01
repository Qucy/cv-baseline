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
        self.conv1 = layers.Conv2D(filter_num, kernel_size=1, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dwConv1 = layers.SeparableConv2D(filter_num, kernel_size=3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.act = layers.ReLU()
        self.dwConv2 = layers.SeparableConv2D(filter_num, kernel_size=3, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.pooling = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')


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
    def __init__(self, num_filters):
        super(XceptionBlock2, self).__init__()
        self.act1 = layers.ReLU()
        self.dwConv1 = layers.SeparableConv2D(num_filters, kernel_size=3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.act2 = layers.ReLU()
        self.dwConv2 = layers.SeparableConv2D(num_filters, kernel_size=3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.act3 = layers.ReLU()
        self.dwConv3 = layers.SeparableConv2D(num_filters, kernel_size=3, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()

    def call(self, inputs, *args, **kwargs):

        x = self.act1(inputs)
        x = self.dwConv1(x)
        x = self.bn1(x)

        x = self.act2(inputs)
        x = self.dwConv2(x)
        x = self.bn2(x)

        x = self.act3(inputs)
        x = self.dwConv3(x)
        x = self.bn3(x)

        x = layers.Add()([inputs, x])

        return x

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
        self.entry_flow_b2 = XceptionBlock1(128)
        self.entry_flow_b3 = XceptionBlock1(256)
        self.entry_flow_b4 = XceptionBlock1(728)

        # middle flow contains 8 block
        self.middle_flow = Sequential([
            XceptionBlock2(728) for _ in range(8)
        ])

        # exit flow
        self.exit_flow_b1 = XceptionBlock1(1024)
        self.exit_flow_b2 = Sequential([
            layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes)
        ])


    def call(self, inputs, training=None, mask=None):
        # (b, 299, 299, 3) -> (b, 149, 149, 64)
        x = self.entry_flow_b1(inputs)
        # (b, 149, 149, 64) -> (b, 75, 75, 128)
        x = self.entry_flow_b2(x)
        # (b, 75, 75, 128) -> (b, 38, 38, 256)
        x = self.entry_flow_b3(x)
        # (b, 38, 38, 256) -> (b, 19, 19, 728)
        x = self.entry_flow_b4(x)
        # (b, 19, 19, 728) -> (b, 19, 19, 728)
        x = self.middle_flow(x)
        # (b, 19, 19, 728) -> (b, 10, 10, 1024)
        x = self.exit_flow_b1(x)
        # (b, 10, 10, 1024) -> (b, num_classes)
        x = self.exit_flow_b2(x)

        return x


if __name__ == '__main__':
    # test forward pass
    fake_images = tf.random.normal([4, 299, 299, 3])
    xception = Xception()
    outputs = xception(fake_images)
    print(outputs.shape)





