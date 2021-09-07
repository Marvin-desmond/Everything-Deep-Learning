"""
# DENSENET
The problems arise with CNNs when they go deeper. This is because the path for
information from the input layer until the output layer (and for the gradient
in the opposite direction) becomes so big, that they can get vanished before
reaching the other side.

The authors solve the problem ensuring maximum information (and gradient) flow.
To do it, they simply connect every layer directly with each other.

Counter-intuitively, by connecting this way DenseNets require fewer parameters
than an equivalent traditional CNN, as there is no need to learn redundant
feature maps.

Another problem with very deep networks was the problem to train, because of
the mentioned flow of information and gradients. DenseNets solve this issue
since each layer has direct access to the gradients from the loss function and
the original input image.

DenseNets do not sum the output feature maps of the layer with the incoming
feature maps but concatenate them.

Since we are concatenating feature maps, this channel dimension is increasing at
every layer. Every layer has access to its preceding feature maps, and therefore,
to the collective knowledge.
"""

import tensorflow as tf

from tensorflow.keras.layers import (
    Conv2D, Dense, MaxPooling2D, BatchNormalization,
    Concatenate, AveragePooling2D, GlobalAveragePooling2D
    )

from tensorflow.keras import Input, Model

import deepplot

class BN_ReLU_Conv(tf.keras.layers.Layer):
    def __init__(self, filters=4, kernel_size=1, strides=1):
        super(BN_ReLU_Conv, self).__init__()
        self.bn = BatchNormalization()
        self.relu = tf.keras.activations.relu
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
    def call(self, in_):
        x = self.bn(in_)
        x = self.relu(x)
        return self.conv(x)

deepplot.Densenetview()


deepplot.Densenetarch()

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(DenseBlock, self).__init__()
        self.layer1 = BN_ReLU_Conv(4 * filters)
        self.layer2 = BN_ReLU_Conv(filters, kernel_size=3)
    def call(self, in_):
        x = self.layer1(in_)
        x = self.layer2(x)
        return x

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(TransitionLayer, self).__init__()
        self.bnReLUConv = BN_ReLU_Conv(filters)
        self.avgpool = AveragePooling2D(pool_size=2, strides=2, padding='same')
    def call(self, x):
        x = self.bnReLUConv(x)
        x = self.avgpool(x)
        return x

deepplot.Densenetarch()

in_ = Input(shape=(224, 224, 3))

x = BN_ReLU_Conv(filters=64, kernel_size=7, strides=2)(in_)
# x = Conv2D(64, 7, strides=2, padding='same')(in_)
x = MaxPooling2D(3, strides=2, padding='same')(x)

for idx, i in enumerate([
    {'filter':32, "repetitions": 6},
    {'filter':32, 'repetitions':12},
    {'filter':32, 'repetitions':24},
    {'filter':32, 'repetitions':16}]):
    aux_in = x
    for _ in range(i['repetitions']):
        x = DenseBlock(i['filter'])(x)
        x = Concatenate()([x, aux_in])
    if idx != 3:
        x = TransitionLayer(i['filter']*2)(x)
x = GlobalAveragePooling2D()(x)
out_ = Dense(1000, activation='softmax')(x)

densenet = Model(inputs=[in_], outputs=[out_])
densenet.summary()
