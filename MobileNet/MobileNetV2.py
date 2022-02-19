import os
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class InvertResidualBlock(layers.Layer):

    def __init__(self, input_channels, output_channels, strides=1, expand_ratio=1, padding='same'):
        """
        initialize method for InvertResidualBlock
        :param filters: number of filters
        :param strides: number of strides for DW Conv
        :param padding: padding for DW Conv
        """
        super(InvertResidualBlock, self).__init__()
        self.pwConv1 = layers.Conv2D(output_channels * expand_ratio, kernel_size=1, strides=1)
        self.bn1 = layers.BatchNormalization()
        self.dwConv = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding=padding)
        self.bn2 = layers.BatchNormalization()
        self.pwConv2 = layers.Conv2D(output_channels, kernel_size=1, strides=1)
        self.bn3 = layers.BatchNormalization()
        self.use_res_connect = (strides == 1 and input_channels == output_channels)

    def call(self, inputs, *args, **kwargs):

        x = self.pwConv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu6(x)
        x = self.dwConv(x)
        x = self.bn2(x)
        x = tf.nn.relu6(x)
        x = self.pwConv2(x)
        x = self.bn3(x)

        if self.use_res_connect:
            return x + inputs
        else:
            return x


class MobileNetV2(Model):

    def __init__(self, num_classes=1000, width_mult=2.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        self.inverted_residual_setting = [
            # t, c, n, s
            # t - expand ratio
            # c - number channels
            # n - number of blocks
            # s - number of stride
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
        self.inverted_residual_blocks = []
        input_channels = 32
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = self._make_divisible(c * width_mult, round_nearest)
            print(input_channels, output_channel)
            for i in range(n):
                strides = s if i == 0 else 1
                self.inverted_residual_blocks.append(InvertResidualBlock(input_channels, output_channel, strides, expand_ratio=t))
                input_channels = output_channel
        self.pooling = layers.AveragePooling2D(7)
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        # b,224,224,3 -> b,112,112,32
        x = self.input_block(inputs)
        # go through inverted residual blocks
        for inverted_residual_block in self.inverted_residual_blocks:
            x = inverted_residual_block(x)
        # b,7,7,c -> b,1,1,c
        x = self.pooling(x)
        # b,1,1,c -> b,1,1,1000
        x = self.fc(x)

        return x

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


if __name__ == '__main__':
    # test forward pass
    fake_images = tf.random.normal([4, 224, 224, 3])
    mobileNetV2 = MobileNetV2()
    outputs = mobileNetV2(fake_images)
    print(outputs.shape)