"""
GoogleNet is a deep convolutional neural network that was
proposed by Szegedy et. al.
The hallmark of this architecture is improved utilization of
computing resources inside the network. This was achieved by
a carefully crafted design that allows for increasing the
depth and width of the network while keeping the computational
budget constant.
"""
import tensorflow as tf
import numpy as np
import deepplot

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Concatenate,
    GlobalAvgPool2D,
    Dropout,
    Flatten,
    Dense,
    Activation
    )

deepplot.InceptionModule()

class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, filters, name=None):
        super(InceptionModule, self).__init__(name=name)
        self.conv1 = Conv2D(filters[0], 1, 1, padding='same', activation='relu')
        self.conv2 = Conv2D(filters[1], 3, 1, padding='same', activation='relu')
        self.conv3 = Conv2D(filters[2], 5, 1, padding='same', activation='relu')
        self.conv4 = Conv2D(filters[3], 1, 1, padding='same', activation='relu')
        self.conv5 = Conv2D(filters[4], 1, 1, padding='same', activation='relu')
        self.conv6 = Conv2D(filters[5], 1, 1, padding='same', activation='relu')
        self.pool = MaxPool2D(3, 1, padding='same')

    def call(self, x):
        path1 = self.conv1(x)

        path2 = self.conv5(x)
        path2 = self.conv2(path2)

        path3 = self.conv6(x)
        path3 = self.conv3(path3)

        path4 = self.pool(x)
        path4 = self.conv4(path4)

        output_layer = Concatenate(axis=-1)([path1, path2, path3, path4])
        return output_layer

"""
One of the main beneficial aspects of this architecture is that it allows
for increasing the number of units at each stage significantly without an
uncontrolled blow-up in computational complexity.
The ubiquitous use of dimension reduction allows for shielding the large
number of input filters of the last stage to the next layer, first reducing
their dimension before convolving over them with a large patch size.
Another practically useful aspect of this design is that it aligns with the
intuition that visual information should be processed at various scales and
then aggregated so that the next stage can abstract features from different
scales simultaneously.
"""
deepplot.GoogleNet()
in_ = Input(shape=(224, 224, 3))
c1 = Conv2D(64, 7, 2, padding='same', activation='relu')(in_)
m1 = MaxPool2D(3, 2, padding='same')(c1)
b1 = tf.keras.layers.BatchNormalization()(m1)
c2 = Conv2D(64, 1, 1, padding='same', activation='relu')(b1)
c3 = Conv2D(192, 3, 1, padding='same', activation='relu')(c2)
b2 = tf.keras.layers.BatchNormalization()(c3)
m2 = MaxPool2D(3, 2, padding='same')(b2)
inc_1 = InceptionModule([64, 128, 32, 32, 96, 16])(m2)
inc_2 = InceptionModule([128, 192, 96, 64, 128, 32])(inc_1)
m3 = MaxPool2D(3, 2, padding='same')(inc_2)
inc_3 = InceptionModule([192, 208, 48, 64, 96, 16])(m3)
inc_4 = InceptionModule([160, 224, 64, 64, 112, 24])(inc_3)
inc_5 = InceptionModule([128, 256, 64, 64, 128, 24])(inc_4)
inc_6 = InceptionModule([112, 288, 64, 64, 144, 32])(inc_5)
inc_7 = InceptionModule([256, 320, 128, 128, 160, 32])(inc_6)
m4 = MaxPool2D(3, 2, padding='same')(inc_7)
inc_8 = InceptionModule([256, 320, 128, 128, 160, 32])(m4)
inc_9 = InceptionModule([384, 384, 128, 128, 192, 48])(inc_8)
avg_pool = GlobalAvgPool2D()(inc_9)
drop_out = Dropout(rate=0.4)(avg_pool)
out_ = Dense(1000, activation='softmax')(drop_out)

model = Model(inputs=in_, outputs=out_)

model.summary()

"""
Given the relatively large depth of the network, the ability to
propagate gradients back through all the layers in an effective
manner was a concern. One interesting insight is that the strong
performance of relatively shallower networks on this task suggests
that the features produced by the layers in the middle of the
network should be very discriminative.

By adding auxiliary classifiers connected to these intermediate layers,
we would expect to encourage discrimination in the lower stages in the
classifier, increase the gradient signal that gets propagated back, and
provide additional regularization.

These classifiers take the form of smaller convolutional networks put on
top of the output of the Inception (4a) and (4d) modules. During training,
their loss gets added to the total loss of the network with a discount
weight (the losses of the auxiliary classifiers were weighted by 0.3).
At inference time, these auxiliary networks are discarded.
"""

deepplot.GoogleNetAux()

class auxiliaryLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(auxiliaryLayer, self).__init__(name=name)
        self.avg_pool = tf.keras.layers.AvgPool2D(5, 3, padding='valid')
        self.conv = Conv2D(1, 1, padding='same', activation='relu')
        self.flatten = Flatten()
        self.dense_1 = Dense(1000, activation='relu')
        self.dense_2 = Dense(1000, activation='softmax')
    def call(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        out = self.dense_2(x)
        return out



in_ = Input(shape=(224, 224, 3))
c1 = Conv2D(64, 7, 2, padding='same', activation='relu')(in_)
m1 = MaxPool2D(3, 2, padding='same')(c1)
b1 = tf.keras.layers.BatchNormalization()(m1)
c2 = Conv2D(64, 1, 1, padding='same', activation='relu')(b1)
c3 = Conv2D(192, 3, 1, padding='same', activation='relu')(c2)
b2 = tf.keras.layers.BatchNormalization()(c3)
m2 = MaxPool2D(3, 2, padding='same')(b2)
inc_1 = InceptionModule([64, 128, 32, 32, 96, 16], name='3a')(m2)
inc_2 = InceptionModule([128, 192, 96, 64, 128, 32], name='3b')(inc_1)
m3 = MaxPool2D(3, 2, padding='same')(inc_2)
inc_3 = InceptionModule([192, 208, 48, 64, 96, 16], name='4a')(m3)
out_1 = auxiliaryLayer()(inc_3)
inc_4 = InceptionModule([160, 224, 64, 64, 112, 24], name='4b')(inc_3)
inc_5 = InceptionModule([128, 256, 64, 64, 128, 24], name='4c')(inc_4)
inc_6 = InceptionModule([112, 288, 64, 64, 144, 32], name='4d')(inc_5)
out_2 = auxiliaryLayer()(inc_6)
inc_7 = InceptionModule([256, 320, 128, 128, 160, 32], name='4e')(inc_6)
m4 = MaxPool2D(3, 2, padding='same')(inc_7)
inc_8 = InceptionModule([256, 320, 128, 128, 160, 32], name='5a')(m4)
inc_9 = InceptionModule([384, 384, 128, 128, 192, 48], name='5b')(inc_8)
avg_pool = GlobalAvgPool2D()(inc_9)
drop_out = Dropout(rate=0.4)(avg_pool)
out_3 = Dense(1000, activation='softmax')(drop_out)

model = Model(inputs=in_, outputs=[out_1, out_2, out_3])

model.summary()

img = np.random.randn(1, 224, 224, 3)

out = model(img)

out[0].shape
out[1].shape
out[2].shape
