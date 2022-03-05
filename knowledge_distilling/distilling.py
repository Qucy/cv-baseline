import tensorflow as tf
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
epochs = 5
IMG_SIZE = 32
T = 5.
lambda_stu = 0.05


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



def distilling_loss(teacher_predictions, student_predictions, labels):
    """
    loss function for knowledge distilling
    :param teacher_predictions: teacher network prediction
    :param student_predictions: student network prediction
    :param labels: ground truth labels
    :return: total loss
    """
    kl_loss = tf.keras.losses.KLDivergence()
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    soft_loss = kl_loss(tf.nn.log_softmax(teacher_predictions/T), tf.nn.log_softmax(student_predictions/T))
    hard_loss = ce_loss(labels, student_predictions)
    total_loss = lambda_stu * hard_loss + (1 - lambda_stu) * T * T * soft_loss
    return total_loss


@tf.function
def train_step(images, labels):

    with tf.GradientTape() as tape:
        teacher_predictions = teacher(images)
        student_predictions = student(images)
        t_loss = distilling_loss(teacher_predictions, student_predictions, labels)

    gradients = tape.gradient(t_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student.trainable_variables))

    train_loss(t_loss)
    train_accuracy(labels, student_predictions)

@tf.function
def t_step(images, labels):
    teacher_predictions = teacher(images)
    student_predictions = student(images)
    t_loss = distilling_loss(teacher_predictions, student_predictions, labels)
    val_loss(t_loss)
    val_accuracy(labels, student_predictions)


if __name__ == '__main__':
    """ Here we train student and teacher network for the same epochs
        to see how good for these 2 networks can be and how many epochs will take
    """
    # load teacher model
    teacher = AttentionResNet18(num_classes)
    teacher.load_weights(teacher_checkpoint_filepath)
    # create student model with customized loss function
    student = VGG16(num_classes)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='validation_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for images, labels in ds_train:
            train_step(images, labels)

        for test_images, test_labels in ds_test:
            t_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Validation Loss: {val_loss.result()}, '
            f'Validation Accuracy: {val_accuracy.result() * 100}'
        )