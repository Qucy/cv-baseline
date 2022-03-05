import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

"""
We use VGG 16 as our student network
"""

class ConvBlocks(layers.Layer):
    """
    Basical conv blocks for VGG
    """
    def __init__(self, num_filters, num_convs):
        super(ConvBlocks, self).__init__()
        self.convBlocks = Sequential([layers.Conv2D(num_filters, kernel_size=3, padding='same', activation=tf.nn.relu)  for _ in range(num_convs)])
        self.maxPooling = layers.MaxPool2D(3, strides=2, padding='same')

    def call(self, inputs, training=None):
        return self.maxPooling(self.convBlocks(inputs))



class VGG16(Model):
    """
    VGG 16
    """
    def __init__(self, num_class):
        super(VGG16, self).__init__()
        self.convBlock1 = ConvBlocks(64, 2)
        self.convBlock2 = ConvBlocks(128, 2)
        self.convBlock3 = ConvBlocks(256, 3)
        self.convBlock4 = ConvBlocks(512, 3)
        self.convBlock5 = ConvBlocks(512, 3)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation=tf.nn.relu)
        self.fc2 = layers.Dense(4096, activation=tf.nn.relu)
        self.fc3 = layers.Dense(num_class)


    def call(self, inputs, training=None):
        x = self.convBlock1(inputs)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.convBlock5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
