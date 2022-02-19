import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'


class ChannelShuffle(layers.Layer):
    """
    Channel Shuffle layer
    """
    def __init__(self, num_output_channels, resolution):
        super(ChannelShuffle, self).__init__()
        self.num_output_channels = num_output_channels
        self.resolution = resolution

    def call(self, inputs, *args, **kwargs):
        x = tf.reshape(inputs, [-1, self.resolution, self.resolution, 2, self.num_output_channels//2])
        x = tf.transpose(x, [0,1,2,4,3])
        x = tf.reshape(x, [-1, self.resolution, self.resolution, self.num_output_channels])
        return x


class ShuffleNetBlock(layers.Layer):
    """
    ShuffleNet basic block
    """
    def __init__(self, num_input_channels, num_output_channels, resolution, num_blocks):
        super(ShuffleNetBlock, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.resolution = resolution
        self.num_blocks = num_blocks

        self.B1ShortcutDWConv = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same')
        self.B1ShortcutBN1 = layers.BatchNormalization()
        self.B1ShortcutPWConv = layers.Conv2D(num_output_channels//2, kernel_size=1, strides=1, padding='same')
        self.B1ShortcutBN2 = layers.BatchNormalization()
        self.B1ShortcutAC = layers.ReLU()

        self.B1PWConv1 = layers.Conv2D(num_output_channels, kernel_size=1, strides=1, padding='same')
        self.B1BN1 = layers.BatchNormalization()
        self.B1AC1 = layers.ReLU()
        self.B1DWConv1 = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same')
        self.B1BN2 = layers.BatchNormalization()
        self.B1PWConv2 = layers.Conv2D(num_output_channels//2, kernel_size=1, strides=1, padding='same')
        self.B1BN3 = layers.BatchNormalization()
        self.B1AC2 = layers.ReLU()

        self.channelShuffle = ChannelShuffle(num_output_channels, resolution)

        self.blocks = [Sequential([
            layers.Conv2D(num_output_channels//2, kernel_size=1, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(num_output_channels//2, kernel_size=1, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ]) for _ in range(num_blocks)]


    def call(self, inputs, *args, **kwargs):
        #======================= shuffle block 1 ============================
        # short cut
        x_left = self.B1ShortcutDWConv(inputs)
        x_left = self.B1ShortcutBN1(x_left)
        x_left = self.B1ShortcutPWConv(x_left)
        x_left = self.B1ShortcutBN2(x_left)
        x_left = self.B1ShortcutAC(x_left)
        # normal path
        x_right = self.B1PWConv1(inputs)
        x_right = self.B1BN1(x_right)
        x_right = self.B1AC1(x_right)
        x_right = self.B1DWConv1(x_right)
        x_right = self.B1BN2(x_right)
        x_right = self.B1PWConv2(x_right)
        x_right = self.B1BN3(x_right)
        x_right = self.B1AC2(x_right)
        # concat
        x = layers.concatenate([x_left, x_right], axis=-1)
        # channel shuffle
        x = self.channelShuffle(x)

        # ======================= shuffle block 2 ============================
        # channel split
        for i in range(self.num_blocks):
            shortcut = x[:,:,:,:self.num_output_channels//2]
            x1 = x[:,:,:,self.num_output_channels//2:]
            x1 = self.blocks[0](x1)
            x = layers.concatenate([shortcut, x1], axis=-1)
            # channel shuffle
            x = self.channelShuffle(x)

        return x


class ShuffleNetV2(Model):
    """
    ShuffleNet V2
    """
    def __init__(self, scale_factor=1, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.input_channels = [24, 116, 232] # number of input channels for stage2, stage3, stage4
        self.output_channels = [116, 232, 464] # number of input channels for stage2, stage3, stage4
        self.resolutions = [28, 14, 7] # image resolution for each stage after downsample
        self.num_blocks = [3, 7, 3] # number of shuffle blocks for stage2, stage3 and stage4
        self.input_blocks = Sequential([
            layers.Conv2D(24, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')
        ])
        self.stages = [
            ShuffleNetBlock(self.input_channels[i], self.output_channels[i], self.resolutions[i], self.num_blocks[i])
            for i in range(3)
        ]
        self.conv5 = layers.Conv2D(num_classes * scale_factor, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
        self.pooling = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)


    def call(self, inputs, training=None, mask=None):
        # (b, 224, 224, 3) -> (b, 56, 56, 24)
        x = self.input_blocks(inputs)
        # go through stage2 to stage4
        for stage in self.stages:
            x = stage(x)
        # (b, 7, 7, C) -> (b, 7, 7, num_classes * scale_factor)
        x = self.conv5(x)
        # (b, 7, 7, num_classes * scale_factor) -> (b, num_classes * scale_factor)
        x = self.pooling(x)
        # (b, num_classes * scale_factor) -> (b, num_classes)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    # test forward pass
    fake_images = tf.random.normal([4, 224, 224, 3])
    shuffleNetV2 = ShuffleNetV2()
    outputs = shuffleNetV2(fake_images)
    print(outputs.shape)
