"""
With ConvNets becoming more of a commodity in the computer vision field,
a number of attempts have been made to improve the original architecture
of Krizhevsky et al. (2012) in a bid to achieve better accuracy.
For instance, the best-performing submissions to the ILSVRC-2013 utilised
smaller receptive window size and smaller stride of the first convolutional
layer. Another line of improvements dealt with training and testing the
networks densely over the whole image and over multiple scales.

In this paper, we address another important aspect of ConvNet architecture
design – its depth. To this end, we fix other parameters of the architecture,
and steadily increase the depth of the network by adding more convolutional
layers, which is feasible due to the use of very small (3 × 3) convolution
filters in all layers.
"""
import tensorflow as tf
import numpy as np
import deepplot

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation

"""
ARCHITECTURE

Input - 224 x 224 RGB image
Preprocessing - subtract mean RGB value from each pixel

Conv layers - small receptive field (3 x 3), one has 1 x 1
            - strides 1
            - padding (same) such that spatial resolution is
            preserved after convolution
Pool layers - max pooling
            - spatial pooling carried out by five max-pooling
              layers
            - 2 x 2 pixel window, with stride 2

All hidden layers equipped with rectification non-linearity (ReLU).
Last layer uses Softmax activation function

The width of conv. layers (the number of channels) is rather small,
starting from 64 in the first layer and then increasing by a factor
of 2 after each max-pooling layer, until it reaches 512.

In spite of a large depth, the number of weights in our nets is
not greater than the number of weights in a more shallow net with
larger conv. layer widths and receptive fields
"""

deepplot.VGG()

in_ = Input(shape=(224, 224, 3))
conv_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu')(in_)
conv_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu')(conv_1)
pool_1 = MaxPool2D(pool_size=2, strides=2)(conv_2)

conv_3 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu')(pool_1)
conv_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu')(conv_3)
pool_2 = MaxPool2D(pool_size=2, strides=2)(conv_4)

conv_5 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu')(pool_2)
conv_6 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu')(conv_5)
conv_7 = Conv2D(filters=256, kernel_size=1, strides=1, padding="same", activation='relu')(conv_6)
pool_3 = MaxPool2D(pool_size=2, strides=2)(conv_7)

conv_8 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu')(pool_3)
conv_9 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu')(conv_8)
conv_10 = Conv2D(filters=512, kernel_size=1, strides=1, padding="same", activation='relu')(conv_9)

pool_4 = MaxPool2D(pool_size=2, strides=2)(conv_10)

conv_11 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu')(pool_4)
conv_12 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu')(conv_11)
conv_13 = Conv2D(filters=512, kernel_size=1, strides=1, padding="same", activation='relu')(conv_12)
pool_5 = MaxPool2D(pool_size=2, strides=2)(conv_13)

flatten = Flatten()(pool_5)

dense_1 = Dense(4096, activation='relu')(flatten)
dense_2 = Dense(4096, activation='relu')(dense_1)
dense_3 = Dense(1000, name='dense3')(dense_2)
out_ = Activation(tf.nn.softmax)(dense_3)

model = Model(inputs = in_, outputs = out_)

model.summary()

"""
Why convolutional layers with 1 x 1 kernels?

Surely these layers cannot capture any features
because the look at only one pixel at a time?

. Although they cannot capture spatial patterns,
they can capture patterns along the depth dimension.
. They are configured to output fewer feature maps
than their inputs, so they serve as bottleneck layers,
meaning they reduce dimensionality. This cuts the
computational cost and the number of parameters,
speeding up training and improving generalization.
"""

"""
It is easy to see that a stack of two 3×3 conv. layers
(without spatial pooling in between) has an effective receptive
field of 5×5; three such layers have a 7 × 7 effective receptive
field.

So what have we gained by using, for instance, a stack of three
3×3 conv. layers instead of a single 7×7 layer?

First, we incorporate three non-linear rectification layers instead
of a single one, which makes the decision function more discriminative.

Second, we decrease the number of parameters: assuming that both the input
and the output of a three-layer 3 × 3 convolution stack has C channels,
the stack is parametrised by 3(3^2C^2) = 27C^2 weights;

at the same time, a single 7 × 7 conv. layer would require 7^2C^2 = 49C^2
parameters, i.e. 81% more.
This can be seen as imposing a regularisation on the 7 × 7 conv. filters,
forcing them to have a decomposition through the 3 × 3 filters (with non-linearity
injected in between).
"""


_, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test.shape, y_test.shape
sample_x = x_test[:1_000]
sample_y = y_test[:1_000]
del x_test, y_test


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for i, img in enumerate(sample_x[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
plt.show()

get_mean = lambda x: np.mean(x, axis=tuple(range(x.ndim-1)))
 # a.reshape(-1,a.shape[-1]).mean(0)
means = [get_mean(img) for img in sample_x[:5]]
scaled_img = [img - mean for img, mean in zip(sample_x[:5],means)]
scaled_img = np.vstack((scaled_img))
scaled_img = scaled_img.reshape(5, 32, 32, 3)

plt.imshow(scaled_img[0].astype("uint8"))
plt.show()


deepplot.VGG()
"""
The initialisation of the network weights is important, since
bad initialisation can stall learning due to the instability
of gradient in deep nets.

To circumvent this problem, we began with training the
configuration A (Table 1), shallow enough to be trained with
random initialisation. Then, when training deeper architectures,
we initialised the first four convolutional layers and the last
three fully connected layers with the layers of net A (the
intermediate layers were initialised randomly). We did not
decrease the learning rate for the pre-initialised layers,
allowing them to change during learning.

"""

from tensorflow.keras.applications.vgg16 import VGG16
vgg_model = VGG16(weights='imagenet', include_top=True)

# initialised the first four convolutional layers
index = 0
for i, layer in enumerate(vgg_model.layers):
    if layer.count_params() > 0:
        model.layers[i].set_weights(layer.get_weights())
        index += 1
    else:
        pass
    if index == 4:
        break

# initialised the last three fully connected layers
index = 0
for i in range(len(vgg_model.layers)):
    i = -1 - i
    # If activation is last layer, then
    # model.layers[i-1]
    # else
    # model.layers[i]
    # if activation within dense layer
    model.layers[i-1].set_weights(vgg_model.layers[i].get_weights())
    print("Done")
    index += 1
    if index == 3:
        break

"""
TRAINING

Optimizer - multinomial logistic (softmax) regression objective
            using mini-batch gradient descent ~ found in
            tf.train.GradientDescentOptimizer() in TF1, in TF2 it
            is replaced by tf.keras.optimizers.SGD()
            BATCH SIZE - 256
            MOMENTUM - 0.9

Learning rate - (initial) 10−2
                decreased by a factor of 10 when the validation
                set accuracy stopped improving.
                Total, decreased 3 times
                LR stopped after 370K iterations (74 epochs).
"""
model.summary()
tf.compat.v1.train.GradientDescentOptimizer

model.compile(
    optimizer = tf.keras.optimizers.SGD(
    learning_rate = 10**-2, momentum = 0.9,
    ),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
)

reduce_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = "val_accuracy",
    factor = 0.1,
    patience = 10,
    cooldown=74,
    min_lr = 10**-2
)


model.fit(sample_x, sample_y, epochs=1, batch_size=16,
        validation_split=0.1, validation_batch_size=16,
        callbacks=[reduce_on_plat])

"""

For random initialisation (where applicable), we sampled the weights
from a normal distribution with the zero mean and 10−2 variance.
The biases were initialised with zero. It is worth noting that
after the paper submission we found that it is possible to initialise
the weights without pre-training by using the random initialisation
procedure of Glorot & Bengio (2010).

Training  - regularisation ~ weight decay (L2) = 5 · 10−4
            dropout ~ first two fully-connected layers = 0.5
"""
conv_layer = Conv2D(
    filters=64, kernel_size=3, strides=1, padding='same',
    kernel_initializer = tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=np.sqrt(10**-2)
    ),
    bias_initializer = 'zeros'
)

dense_layer = Dense(
    units=4,
    kernel_initializer = tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=np.sqrt(10**-2)
    ),
    bias_initializer = tf.keras.initializers.Zeros(),
    activity_regularizer = tf.keras.regularizers.l2(5e-4),
)

drop_out_layer = tf.keras.layers.Dropout(
  rate = 0.5
)
