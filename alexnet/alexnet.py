import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# define hyper parameters
SHUFFLE_SIZE = 60000
BATCH_SIZE = 256
AUTO_TUNE = tf.data.AUTOTUNE
lr = 1e-3

# loading cifar100 datasets
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar100.load_data()
print(f"Retrieved {x_train.shape[0]} training samples and {x_val.shape[0]} testing samples")
print(f"Image shape is {x_train.shape[1:]}")


# define a normalization function
def normalize(x, y):
    x = tf.image.resize(x, (227, 227))
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# generate train and validation datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache().map(normalize).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=AUTO_TUNE)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.cache().map(normalize).prefetch(buffer_size=AUTO_TUNE)


# define model
alexnet = Sequential([
    layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227, 227, 3)),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(100)
])

alexnet.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

print(alexnet.summary())

# we don't need train this alex model because it's an old model have over 58 million parameters (that's too much)
# we can understand the model code and know how to train this model is enough
alexnet.fit(ds_train, validation_data=ds_val, epochs=10)
