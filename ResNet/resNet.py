import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class ResidualBlock(layers.Layer):

    def __init__(self, filter_num, strides=1):
        super(ResidualBlock, self).__init__()
        self.c1 = layers.Conv2D(filter_num, kernel_size=3, strides=strides, padding='same')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.LeakyReLU(.2)

        self.c2 = layers.Conv2D(filter_num, kernel_size=3, strides=1, padding='same')
        self.b2 = layers.BatchNormalization()

        if strides > 1:
            self.downSample = Sequential()
            self.downSample.add(layers.Conv2D(filter_num, kernel_size=1, strides=strides))
        else:
            self.downSample = lambda x : x


    def call(self, inputs, training=None):

        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)

        short_cut = self.downSample(inputs)

        x = layers.add([x, short_cut])
        x = tf.nn.relu(x)

        return x




class ResNet(keras.Model):

    def __init__(self, layer_dims, num_classes):
        """
        init function for ResNet
        :param layer_dims: [2,2,2,2] => how many residual blocks for each residual module
        :param num_classes: number of classes
        """
        super(ResNet, self).__init__()
        self.input_layer = Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(.2),
            layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        ])

        self.resBlock1 = self.buildResidualBlock(64, layer_dims[0])
        self.resBlock2 = self.buildResidualBlock(128, layer_dims[1], strides=2)
        self.resBlock3 = self.buildResidualBlock(256, layer_dims[2], strides=2)
        self.resBlock4 = self.buildResidualBlock(512, layer_dims[3], strides=2)

        # [b, 512, h, w] => [b, 512, 1, 1]
        self.avgPool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)



    def call(self, inputs, training=None):

        x = self.input_layer(inputs)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)

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




def ResNet18():
    return ResNet([2, 2, 2, 2], 100)

def ResNet34():
    return ResNet([3, 4, 6, 3], 100)



def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# init hyper parameter
batch_size = 256
AUTO_TUNE = tf.data.AUTOTUNE
lr = 1e-4

# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# create datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache().map(preprocess).shuffle(50000).batch(batch_size).prefetch(buffer_size=AUTO_TUNE)

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.cache().batch(batch_size).map(preprocess)

resnet18 = ResNet18()

resnet18.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# callback for early stop
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# train epoch 10
# 196/196 [==============================] - 48s 241ms/step - loss: 0.0294 - accuracy: 0.9972 - val_loss: 2.7383 - val_accuracy: 0.4228
resnet18.fit(ds_train, validation_data=ds_test, callbacks=[callback], epochs=100)
# eval
#resnet18.evaluate(ds_test)
# save weights
# resnet18.save_weights('./models/cifar100_resnet18/resnet18.ckpt')






