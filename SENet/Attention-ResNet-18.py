import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow import keras

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

"""
Here we are going to build a ResNet and integrated with any attention modules you want
"""
class ResidualBlock(layers.Layer):

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

#====================================================================================
#===============================SE Attention=========================================
#====================================================================================
class SEBlock(layers.Layer):
    """
    SE attention block
    Global Average Pooling -> Fully connected(channel//reduction) -> Fully connected(channel) -> Sigmoid * inputs
    """
    def __init__(self, filter_nums, reduction_rate=16):
        super(SEBlock, self).__init__()
        self.p1 = layers.GlobalAveragePooling2D()
        self.r1 = layers.Reshape((1, 1, filter_nums))
        self.d1 = layers.Dense(filter_nums//reduction_rate,
                               kernel_initializer='he_normal',
                               use_bias=False,
                               activation=tf.nn.relu,
                               bias_initializer='zeros')
        self.d2 = layers.Dense(filter_nums,
                               kernel_initializer='he_normal',
                               use_bias=False,
                               activation=tf.nn.relu,
                               bias_initializer='zeros')


    def call(self, inputs, *args, **kwargs):
        x = self.p1(inputs)
        x = self.r1(x)
        x = self.d1(x)
        x = self.d2(x)
        attention = tf.nn.sigmoid(x)
        return layers.multiply([inputs, attention])


#====================================================================================
#===============================CBAM Attention=======================================
#====================================================================================
class CBAMChannelBlock(layers.Layer):
    """
    CBAM channel attention block
    Global Average pool + Global Max pool -> shared FC layers -> add -> sigmoid * inputs
    """
    def __init__(self, filter_nums, reduction_rate=8):
        super(CBAMChannelBlock, self).__init__()
        self.avgPool = layers.GlobalAveragePooling2D()
        self.maxPool = layers.GlobalMaxPooling2D()
        self.r1 = layers.Reshape((1, 1, filter_nums))
        self.d1 = layers.Dense(filter_nums//reduction_rate,
                               kernel_initializer='he_normal',
                               use_bias=False,
                               activation=tf.nn.relu,
                               bias_initializer='zeros')
        self.d2 = layers.Dense(filter_nums,
                               kernel_initializer='he_normal',
                               use_bias=False,
                               activation=tf.nn.relu,
                               bias_initializer='zeros')


    def call(self, inputs, *args, **kwargs):

        avg_pool = self.avgPool(inputs)
        max_pool = self.maxPool(inputs)

        avg_pool = self.r1(avg_pool)
        max_pool = self.r1(max_pool)

        avg_pool = self.d1(avg_pool)
        max_pool = self.d1(max_pool)

        avg_pool = self.d2(avg_pool)
        max_pool = self.d2(max_pool)

        channel_attention = layers.Add()([avg_pool, max_pool])
        channel_attention = tf.nn.sigmoid(channel_attention)

        return layers.multiply([inputs, channel_attention])


class CBAMSpatialBlock(layers.Layer):
    """
    CBAM spatial attention block
    Average pool + Max pool -> concat -> conv layers -> sigmoid * inputs
    """
    def __init__(self):
        super(CBAMSpatialBlock, self).__init__()
        self.avgPool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))
        self.maxPool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))
        self.conv = layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)


    def call(self, inputs, *args, **kwargs):

        avg_pool = self.avgPool(inputs)
        max_pool = self.maxPool(inputs)

        spatial_attention = layers.concatenate([avg_pool, max_pool], axis=-1)
        spatial_attention = self.conv(spatial_attention)

        spatial_attention = tf.nn.sigmoid(spatial_attention)

        return layers.multiply([inputs, spatial_attention])


class CBAMBlock(layers.Layer):

    def __init__(self, filter_nums, reduction_rate=8):
        super(CBAMBlock, self).__init__()
        self.channelAttention = CBAMChannelBlock(filter_nums, reduction_rate)
        self.spatialAttention = CBAMSpatialBlock()

    def call(self, inputs, *args, **kwargs):
        cbam_feature = self.channelAttention(inputs)
        cbam_feature = self.spatialAttention(cbam_feature)
        return cbam_feature


#====================================================================================
#===========================ResNet with Attention====================================
#====================================================================================
class AttentionResNet(Model):

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

        self.resBlock1 = self.buildResidualBlock(64, layer_dims[0])
        #self.seBlock1 = SEBlock(64)
        self.cbamBlock1 = CBAMBlock(64)

        self.resBlock2 = self.buildResidualBlock(128, layer_dims[1], strides=2)
        #self.seBlock2 = SEBlock(128)
        self.cbamBlock2 = CBAMBlock(128)

        self.resBlock3 = self.buildResidualBlock(256, layer_dims[2], strides=2)
        #self.seBlock3 = SEBlock(256)
        self.cbamBlock3 = CBAMBlock(256)

        self.resBlock4 = self.buildResidualBlock(512, layer_dims[3], strides=2)
        #self.seBlock4 = SEBlock(512)
        self.cbamBlock4 = CBAMBlock(512)

        # [b, 512, h, w] => [b, 512, 1, 1]
        self.avgPool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)


    def call(self, inputs, *args, **kwargs):
        x = self.input_layer(inputs)

        x = self.resBlock1(x)
        #x = self.seBlock1(x)
        x = self.cbamBlock1(x)

        x = self.resBlock2(x)
        #x = self.seBlock2(x)
        x = self.cbamBlock2(x)

        x = self.resBlock3(x)
        #x = self.seBlock3(x)
        x = self.cbamBlock3(x)

        x = self.resBlock4(x)
        #x = self.seBlock4(x)
        x = self.cbamBlock4(x)

        x = self.avgPool(x)
        x = self.fc(x)
        return x


    def buildResidualBlock(self, filter_number, blocks, strides=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(ResidualBlock(filter_number, strides))
        for _ in range(1, blocks):
            res_blocks.add(ResidualBlock(filter_number, strides=1))

        return res_blocks


# init hyper parameter
batch_size = 256
AUTO_TUNE = tf.data.AUTOTUNE
lr = 1e-4
num_classes = 10

def AttentionResNet18():
    return AttentionResNet([2, 2, 2, 2], num_classes)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# create datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache().map(preprocess).shuffle(50000).batch(batch_size).prefetch(buffer_size=AUTO_TUNE)

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.cache().batch(batch_size).map(preprocess)

attentionResNet18 = AttentionResNet18()

attentionResNet18.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

# callback for early stop
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# train epoch 5
attentionResNet18.fit(ds_train, validation_data=ds_test, callbacks=[callback], epochs=5)

# result CIFAR10
# ResNet18 after 5 epochs -> total 44s 221ms/step - train loss: 0.1360 - train accuracy: 0.9555 - val_loss: 1.7035 - val_accuracy: 0.6153
# ResNet18 with SE attention after 5 epochs -> total 50s 254ms/step - train loss: 0.1704 - train accuracy: 0.9447 - val_loss: 1.2722 - val_accuracy: 0.6767
# ResNet18 with CBAM attention after 5 epochs ->

# result CIFAR100
# ResNet18 after 5 epochs -> total 45s 231ms/step - train loss: 1.1268 - train accuracy: 0.7059 - val_loss: 2.4902 - val_accuracy: 0.3889
# ResNet18 with SE attention after 5 epochs -> total 52s 264ms/step - train loss: 1.4035 - train accuracy: 0.6179 - val_loss: 2.5279 - val_accuracy: 0.3805
# ResNet18 with CBAM attention after 5 epochs -> total 56s 285ms/step - train loss: 1.5187 - train accuracy: 0.5837 - val_loss: 2.4526 - val_accuracy: 0.3764