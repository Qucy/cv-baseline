import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

"""
use attention residual network as teacher network
"""

class ResidualBlock(layers.Layer):
    """
    Residual block
    """
    def __init__(self, filter_num, strides=1):
        super(ResidualBlock, self).__init__()
        self.c1 = layers.Conv2D(filter_num, kernel_size=3, strides=strides, padding='same')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.LeakyReLU(.2)

        self.c2 = layers.Conv2D(filter_num, kernel_size=3, strides=1, padding='same')
        self.b2 = layers.BatchNormalization()

        if strides > 1:
            self.downSample = layers.Conv2D(filter_num, kernel_size=1, strides=strides)
        else:
            self.downSample = lambda x : x


    def call(self, inputs, *args, **kwargs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        shortcut = self.downSample(inputs)
        x = layers.add([x, shortcut])
        x = tf.nn.relu(x)
        return x


class ECABlock(layers.Layer):
    """
    ECA attention block
    Global Average pool + Global Max pool -> shared FC layers -> add -> sigmoid * inputs
    """
    def __init__(self, filter_nums, b=1, gamma=2):
        super(ECABlock, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.kernel_size = int(abs((tf.math.log(tf.cast(filter_nums, dtype=tf.float32), 2) + b) / gamma))
        self.c1 = layers.Conv1D(1, kernel_size=self.kernel_size, padding="same", use_bias=False)


    def call(self, inputs, *args, **kwargs):
        attention = self.avg(inputs)
        attention = layers.Reshape((-1, 1))(attention)
        attention = self.c1(attention)
        attention = tf.nn.sigmoid(attention)
        attention = layers.Reshape((1, 1, -1))(attention)

        return layers.Multiply()([inputs, attention])



class AttentionResNet(Model):
    """
    Attention Residual Network
    """
    def __init__(self, layer_dims, num_classes):
        """
        init function for SEResNet
        :param layer_dims: [2,2,2,2] => how many residual blocks for each residual module
        :param num_classes: number of classes
        """
        super(AttentionResNet, self).__init__()
        self.input_layer = Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(.2),
            layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        ])

        self.resBlock1 = self._buildResidualBlock(64, layer_dims[0])
        self.ecaBlock1 = ECABlock(64)

        self.resBlock2 = self._buildResidualBlock(128, layer_dims[1], strides=2)
        self.ecaBlock2 = ECABlock(128)

        self.resBlock3 = self._buildResidualBlock(256, layer_dims[2], strides=2)
        self.ecaBlock3 = ECABlock(256)

        self.resBlock4 = self._buildResidualBlock(512, layer_dims[3], strides=2)
        self.ecaBlock4 = ECABlock(512)

        # [b, 512, h, w] => [b, 512, 1, 1]
        self.avgPool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)


    def call(self, inputs, *args, **kwargs):
        x = self.input_layer(inputs)

        x = self.resBlock1(x)
        x = self.ecaBlock1(x)

        x = self.resBlock2(x)
        x = self.ecaBlock2(x)

        x = self.resBlock3(x)
        x = self.ecaBlock3(x)

        x = self.resBlock4(x)
        x = self.ecaBlock4(x)

        x = self.avgPool(x)
        x = self.fc(x)
        return x


    def _buildResidualBlock(self, filter_number, blocks, strides=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(ResidualBlock(filter_number, strides))
        for _ in range(1, blocks):
            res_blocks.add(ResidualBlock(filter_number, strides=1))

        return res_blocks


def AttentionResNet18(num_classes):
    return AttentionResNet([2, 2, 2, 2], num_classes)



