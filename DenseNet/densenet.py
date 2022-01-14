import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class ConvBlock(layers.Layer):
    """
    A convolution block in dense net
    BN + RELU + CONV + BN + RELU + CONV
    """

    def __init__(self, growth_rate, name):
        super(ConvBlock, self).__init__()
        self.b1 = layers.BatchNormalization(name=name + '_0_bn')
        self.r1 = layers.ReLU(name=name + '_0_relu')
        self.c1 = layers.Conv2D(4 * growth_rate, kernel_size=1, use_bias=False, name=name + '_1_conv')
        self.b2 = layers.BatchNormalization(name=name + '_1_bn')
        self.r2 = layers.ReLU(name=name + '_1_relu')
        self.c2 = layers.Conv2D(growth_rate, kernel_size=3,  padding='same', use_bias=False, name=name + '_2_conv')
        self.concat = layers.Concatenate(axis=3, name=name + '_concat')


    def call(self, inputs, training=None):

        x = self.b1(inputs, training=training)
        x = self.r1(x)
        x = self.c1(x)
        x = self.b2(x, training=training)
        x = self.r2(x)
        x = self.c2(x)
        outputs = self.concat([x, inputs])

        return outputs


class DenseBlock(layers.Layer):
    """
    A dense block
    """
    def __init__(self, blocks, name):
        super(DenseBlock, self).__init__()
        self.denseBlocks = Sequential([
            ConvBlock(growth_rate=32, name=name + '_block' + str(i + 1)) for i in range(blocks)
        ])

    def call(self, inputs, training=None):
        return self.denseBlocks(inputs, training=training)


class TransistionBlock(layers.Layer):
    """
    A Transistion block
    """
    def __init__(self, num_filters, name):
        super(TransistionBlock, self).__init__()
        self.b1 = layers.BatchNormalization(name= name + '_bn')
        self.r1 = layers.ReLU(name= name + '_relu')
        self.c1 = layers.Conv2D(num_filters, kernel_size=1, use_bias=False, name= name + '_conv')
        self.p1 = layers.AveragePooling2D(pool_size=2, strides=2, name= name + '_pool')


    def call(self, inputs, training=None):
        x = self.b1(inputs)
        x = self.r1(x)
        x = self.c1(x)
        x = self.p1(x)
        return x


class DenseNet(Model):

    def __init__(self, blocks, num_filters, num_classes):
        super(DenseNet, self).__init__()
        self.input_layer = Sequential([
             layers.ZeroPadding2D((3, 3)),
             layers.Conv2D(64, kernel_size=7, strides=2, use_bias=False, name='conv1/conv'),
             layers.BatchNormalization(name='conv1/bn'),
             layers.ReLU(name='conv1/relu')]
        )
        self.p1 = layers.ZeroPadding2D((1, 1))
        self.p2 = layers.MaxPool2D(3, strides=2, name='pool1')

        self.d1 = DenseBlock(blocks[0], name='conv2')
        self.t1 = TransistionBlock(num_filters[0], name='pool2')

        self.d2 = DenseBlock(blocks[1], name='conv3')
        self.t2 = TransistionBlock(num_filters[1], name='pool3')

        self.d3 = DenseBlock(blocks[2], name='conv4')
        self.t3 = TransistionBlock(num_filters[2], name='pool4')

        self.d4 = DenseBlock(blocks[3], name='conv5')

        self.bn = layers.BatchNormalization(name='bn')
        self.relu = layers.ReLU(name='relu')

        self.avg_pool = layers.GlobalAveragePooling2D(name='avg_pool')

        self.fc = layers.Dense(num_classes, name='fc'+str(num_classes))


    def call(self, inputs, training=None):
        # 224, 224, 3 => 112, 112, 64
        x = self.input_layer(inputs)

        # 112, 112, 64 => 56, 56, 64
        x = self.p1(x)
        x = self.p2(x)

        # 56,56,64 -> 56,56,64+32*block[0]
        # Densenet121 56,56,64 -> 56,56,64+32*6 == 56,56,256
        x = self.d1(x)

        # 56,56,64+32*block[0] -> 28,28,32+16*block[0]
        # Densenet121 56,56,256 -> 28,28,32+16*6 == 28,28,128
        x = self.t1(x)

        # 28,28,32+16*block[0] -> 28,28,32+16*block[0]+32*block[1]
        # Densenet121 28,28,128 -> 28,28,128+32*12 == 28,28,512
        x = self.d2(x)

        # Densenet121 28,28,512 -> 14,14,256
        x = self.t2(x)

        # Densenet121 14,14,256 -> 14,14,256+32*block[2] == 14,14,1024
        x = self.d3(x)

        # Densenet121 14,14,1024 -> 7,7,512
        x = self.t3(x)

        # Densenet121 7,7,512 -> 7,7,512+32*block[2] == 7,7,1024
        x = self.d4(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.fc(x)

        return x



def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# init hyper parameter
batch_size = 256
AUTO_TUNE = tf.data.AUTOTUNE
lr = 1e-4
num_classes = 100

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

denseNet121 = DenseNet([6, 12, 24, 16], [128, 256, 512],num_classes=num_classes)
#denseNet169 = DenseNet([6, 12, 32, 32], num_classes=num_classes)
#denseNet201 = DenseNet([6, 12, 48, 32], num_classes=num_classes)

denseNet121.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])


# callback for early stop
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# train Epoch 20 196/196 [==============================] - 23s 114ms/step - loss: 0.1248 - accuracy: 0.9783 - val_loss: 3.8949 - val_accuracy: 0.2677
denseNet121.fit(ds_train, validation_data=ds_test, callbacks=[callback], epochs=100)
# looks like DenseNet is quite easy to overfitting, need data argumentation definitely


# you can also download pre-trained model
# tf.keras.applications.densenet.DenseNet121(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000
# )
# tf.keras.applications.densenet.DenseNet201(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000
# )
# tf.keras.applications.densenet.DenseNet201(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000
# )










