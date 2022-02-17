import os
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class DWConvBlock(layers.Layer):
    
    def __init__(self, filters, strides=1, padding='valid'):
        """
        initialize method for depthwise separable conv layers
        :param filters: number of filters
        :param strides: number of strides for DW Conv
        :param padding: padding for DW Conv
        """
        super(DWConvBlock, self).__init__()
        self.dwConv = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding=padding)
        self.bn1 = layers.BatchNormalization()
        self.pwConv = layers.Conv2D(filters, kernel_size=1, strides=1)
        self.bn2 = layers.BatchNormalization()


    def call(self, inputs, *args, **kwargs):

        x = self.dwConv(inputs)
        x = self.bn1(x)
        x = tf.nn.relu6(x)

        x = self.pwConv(x)
        x = self.bn2(x)
        x = tf.nn.relu6(x)

        return x



class MobileNetV1(Model):
    
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.input_block = Sequential([
            layers.Conv2D(32, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.DWBlock1 = DWConvBlock(filters=64, strides=1, padding='same')
        self.DWBlock2 = DWConvBlock(filters=64, strides=2, padding='same')
        self.DWBlock3 = DWConvBlock(filters=128, strides=1, padding='same')
        self.DWBlock4 = DWConvBlock(filters=128, strides=2, padding='same')
        self.DWBlock5 = DWConvBlock(filters=256, strides=1, padding='same')
        self.DWBlock6 = DWConvBlock(filters=256, strides=2, padding='same')
        self.DWFiveBlocks = Sequential([
            DWConvBlock(filters=512, strides=1, padding='same') for _ in range(5)
        ])
        self.DWBlock7 = DWConvBlock(filters=512, strides=2, padding='same')
        self.DWBlock8 = DWConvBlock(filters=1024, strides=1, padding='same')
        self.pooling = layers.AveragePooling2D(7)
        self.fc = layers.Dense(num_classes)


    def call(self, inputs, training=None, mask=None):
        # b,224,224,3 -> b,112,112,32
        x = self.input_block(inputs)
        # b,112,112,32 -> b,112,112,64
        x = self.DWBlock1(x)
        # b,112,112,64 -> b,56,56,64
        x = self.DWBlock2(x)
        # b,56,56,64 -> b,56,56,128
        x = self.DWBlock3(x)
        # b,56,56,64 -> b,28,28,128
        x = self.DWBlock4(x)
        # b,28,28,128 -> b,28,28,256
        x = self.DWBlock5(x)
        # b,28,28,256 -> b,14,14,256
        x = self.DWBlock6(x)
        # b,14,14,256 -> b,14,14,512
        x = self.DWFiveBlocks(x)
        # b,14,14,512 -> b,7,7,512
        x = self.DWBlock7(x)
        # b,7,7,512 -> b,7,7,1024
        x = self.DWBlock8(x)
        # b,7,7,1024 -> b,1,1,1024
        x = self.pooling(x)
        # b,1,1,1024 -> b,1,1,1000
        x = self.fc(x)

        return x



if __name__ == '__main__':
    # test forward pass
    fake_images = tf.random.normal([4, 224, 224, 3])
    mobileNetV1 = MobileNetV1()
    outputs = mobileNetV1(fake_images)
    print(outputs.shape)