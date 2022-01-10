import os
import tensorflow as tf
from tensorflow import keras

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

# hyper parameter
num_classes = 10
batch_size = 32
lr = 1e-4
AUTO_TUNE = tf.data.AUTOTUNE

def normalize(x, y):
    x = tf.image.resize(x, (150, 150))
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


# For InceptionV2 and InceptionV3, i recommend to use inceptionV3 directly, since V3 performance is better
# For InceptionV3, we don need to write the model architecture, we can directly use pre-trained model directly
# Set include_top=False and use transfer learning to train on our datasets
# We can also set trainable=True, if we want to fine tune other layer in the model
inceptionV3 = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None)

# use this one if you want to train inceptionResNetV2
# inceptionResNetV2 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None)

# Freeze base model
inceptionV3.trainable = False


# Create new model on top.
inputs = keras.Input(shape=(150, 150, 3)) # define your image size here
x = inceptionV3(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(num_classes)(x) # input your number of class here
model = keras.Model(inputs, outputs)


loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

# Iterate over the batches of a dataset.
for inputs, targets in ds_train:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the *trainable* weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))