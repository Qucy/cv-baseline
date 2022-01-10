import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class InceptionModule(layers.Layer):

    def __init__(self, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool):
        super(InceptionModule, self).__init__()
        self.conv1x1 = layers.Conv2D(filters_1x1, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
        self.conv3x3_reduce = layers.Conv2D(filters_3x3_reduce, kernel_size=1, strides=1, activation=tf.nn.relu)
        self.conv3x3 = layers.Conv2D(filters_3x3, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv5x5_reduce = layers.Conv2D(filters_5x5_reduce, kernel_size=1, strides=1, activation=tf.nn.relu)
        self.conv5x5 = layers.Conv2D(filters_5x5, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
        self.pooling = layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        self.pool_proj = layers.Conv2D(filters_pool, kernel_size=1, strides=1, activation=tf.nn.relu)



    def call(self, inputs, *args, **kwargs):

        p1 = self.conv1x1(inputs)

        p2 = self.conv3x3_reduce(inputs)
        p2 = self.conv3x3(p2)

        p3 = self.conv5x5_reduce(inputs)
        p3 = self.conv5x5(p3)

        p4 = self.pooling(inputs)
        p4 = self.pool_proj(p4)

        return layers.concatenate([p1, p2, p3, p4])




class InceptionV1(keras.Model):

    def __init__(self, num_classes):
        super(InceptionV1, self).__init__()
        self.c1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=(224, 224, 3), activation=tf.nn.relu)
        self.p1 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.c2 = layers.Conv2D(192, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.c3 = layers.Conv2D(192, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.inception_3a = InceptionModule(filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool=32)
        self.inception_3b = InceptionModule(filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool=64)
        self.p2 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.inception_4a = InceptionModule(filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool=64)
        self.inception_4b = InceptionModule(filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool=64)
        self.inception_4c = InceptionModule(filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool=64)
        self.inception_4d = InceptionModule(filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool=64)
        self.inception_4e = InceptionModule(filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool=128)
        self.p3 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.inception_5a = InceptionModule(filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool=128)
        self.inception_5b = InceptionModule(filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool=128)
        self.p4 = layers.AveragePooling2D(pool_size=7, strides=1)
        self.dropout = layers.Dropout(0.4)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes)


    def call(self, inputs, training=None):
        # (224, 224, 3) => (112, 112, 64)
        x = self.c1(inputs)
        # (112, 112, 64) => (56, 56, 64)
        x = self.p1(x)
        # (56, 56, 64) => (56, 56, 192)
        x = self.c2(x)
        # (56, 56, 192) => (28, 28, 192)
        x = self.c3(x)
        # (28, 28, 192) => (28, 28, 256)
        x = self.inception_3a(x)
        # (28, 28, 256) => (28, 28, 480)
        x = self.inception_3b(x)
        # (28, 28, 480) => (14, 14, 480)
        x = self.p2(x)
        # (14, 14, 480) => (14, 14, 512)
        x = self.inception_4a(x)
        # (14, 14, 512) => (14, 14, 512)
        x = self.inception_4b(x)
        # (14, 14, 512) => (14, 14, 512)
        x = self.inception_4c(x)
        # (14, 14, 512) => (14, 14, 528)
        x = self.inception_4d(x)
        # (14, 14, 528) => (14, 14, 832)
        x = self.inception_4e(x)
        # (14, 14, 528) => (7, 7, 832)
        x = self.p3(x)
        # (7, 7, 832) => (7, 7, 832)
        x = self.inception_5a(x)
        # (7, 7, 832) => (7, 7, 1024)
        x = self.inception_5b(x)
        # (7, 7, 1024) => (1, 1, 1024)
        x = self.p4(x)
        # (1, 1, 1024) => (1, 1, 1024)
        x = self.dropout(x, training)
        # (1, 1, 1024) => (1024)
        x = self.flatten(x)
        # (024) => (num_classes)
        x = self.dense(x)

        return x


# hyper parameter
num_classes = 10
batch_size = 32
lr = 1e-4
AUTO_TUNE = tf.data.AUTOTUNE


def normalize(x, y):
    x = tf.image.resize(x, (224, 224))
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# loading data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert x_test.shape == (10000, 32, 32, 3)
assert y_test.shape == (10000, 1)
# create data set
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache().map(normalize).shuffle(50000).batch(batch_size).prefetch(buffer_size=AUTO_TUNE)

ds_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_val = ds_val.cache().map(normalize).batch(batch_size).prefetch(buffer_size=AUTO_TUNE)

# inception V1 has too many parameters, i would suggest not train it locally, even my PC with GeFore 1070 8G memory will OOM
# for simplicity we didn't implement auxiliary loss layer here
inceptionV1 = InceptionV1(num_classes)
inceptionV1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
inceptionV1.fit(ds_train, validation_data=ds_val, epochs=2)


