import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# hyper parameter
SHUFFLE_SIZE = 60000
BATCH_SIZE = 256
AUTO_TUNE = tf.data.AUTOTUNE
lr=1e-4


class ConvBlocks(layers.Layer):

    def __init__(self, num_filters, num_convs):
        super(ConvBlocks, self).__init__()
        self.convBlocks = Sequential([layers.Conv2D(num_filters, kernel_size=3, padding='same', activation=tf.nn.relu)  for _ in range(num_convs)])
        self.maxPooling = layers.MaxPool2D(3, strides=2, padding='same')

    def call(self, inputs, training=None):
        return self.maxPooling(self.convBlocks(inputs))



class VGG16(keras.Model):

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


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# loading datasets
(x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

# generate datasets for training and validation
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache().map(preprocess).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=AUTO_TUNE)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.cache().map(preprocess).batch(BATCH_SIZE).prefetch(buffer_size=AUTO_TUNE)

# build model
vgg16 = VGG16(10)

# compile model
vgg16.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# fit
vgg16.fit(ds_train, validation_data=ds_val, epochs=10)
# 10 epochs 196/196 [==============================] - 21s 104ms/step - loss: 0.5032 - accuracy: 0.8211 - val_loss: 0.9013 - val_accuracy: 0.7165