import tensorflow as tf
from tensorflow import keras
from teacher import AttentionResNet18
from student import VGG16

# configurations
teacher_checkpoint_filepath = './model/teacher/t'
student_checkpoint_filepath = './model/student/s'

# init hyper parameter
batch_size = 256
AUTO_TUNE = tf.data.AUTOTUNE
lr = 1e-4
num_classes = 100
epochs = 100
IMG_SIZE = 32


def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = tf.cast(image, dtype=tf.float32) / 255.
  label = tf.cast(label, dtype=tf.int32)
  return image, label

def augment(image_label, seed):
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed.
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size.
  image = tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness.
  image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label


# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# Create a generator.
rng = tf.random.Generator.from_seed(123, alg='philox')

# Create a wrapper function for updating seeds.
def f(x, y):
  seed = rng.make_seeds(2)[0]
  image, label = augment((x, y), seed)
  return image, label

# create datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache().map(f).shuffle(50000).batch(batch_size).prefetch(buffer_size=AUTO_TUNE)

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.cache().batch(batch_size).map(resize_and_rescale)

# create teacher and student network
teacher = AttentionResNet18(num_classes)
student = VGG16(num_classes)

# compile teacher and student network
teacher.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

student.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


# callback for early stop
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# callback for save checkpoint
teacher_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=teacher_checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
student_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=student_checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# callback function for to store tensorboard logs
teacher_log_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./model/teacher/logs/',
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None)
student_log_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./model/student/logs/',
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None)


if __name__ == '__main__':
    """ Here we train student and teacher network for the same epochs
        to see how good for these 2 networks can be and how many epochs will take
    """
    # train teacher or student network
    train_teacher = False

    if train_teacher:
        # train teacher only
        teacher.fit(ds_train, validation_data=ds_test, callbacks=[callback, teacher_checkpoint_callback, teacher_log_callback], epochs=epochs)
    else:
        # train student only
        student.fit(ds_train, validation_data=ds_test, callbacks=[callback, student_checkpoint_callback, student_log_callback], epochs=epochs)


