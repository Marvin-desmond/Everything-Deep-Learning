import tensorflow as tf
import deepplot
import numpy as np

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, AvgPool2D

"""
LeNet-5 is one of the simplest architectures,proposed
by Yann Lecun et. al. in 1989.
It has 2 convolutional and 3 fully-connected layers.
The average-pooling layer as we know it now was called
a sub-sampling layer and it had trainable weights
(nowadays sub-sampling layers do not have trainable
weights)
It has about 60_000 parameters
"""

deepplot.LeNet5()

model = Sequential([
    Input(shape=(32, 32, 1)),
    Conv2D(6, 5, 1, activation='tanh'),
    AvgPool2D(2, 2),
    Conv2D(16, 5, 1, activation='tanh'),
    AvgPool2D(2, 2),
    Flatten(),
    Dense(120, activation='tanh'),
    Dense(84, activation='tanh'),
    Dense_RBF(10, 0.5)
])

K = tf.keras.backend
class Dense_RBF(tf.keras.layers.Layer):
    def __init__(self, units, gamma):
        super(Dense_RBF, self).__init__()
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(
            name = 'mu',
            shape = (int(input_shape[1]), self.units),
            initializer='uniform',
            trainable=True
            )
        super(Dense_RBF, self).build(input_shape)
    def call(self, x):
        diff = K.expand_dims(x) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

model.summary()

in_ = np.random.randn(1, 32, 32, 1)
out_ = model(in_)
out_
"""
The AlexNet CNN architecture won the 2012 ImageNet ILSVRC challenge
by a large margin: it achieved a top-five error rate of 17%, while
the second best achieved only 26%! It was developed by Alex Krizhevsky
(hence the name), Ilya Sutskever, and Geoffrey Hinton. It is similar
to LeNet-5, only much larger and deeper, and it was the first to stack
convolutional layers directly on top of one another, instead of stacking
a pooling layer on top of each convolutional layer.

With 60M parameters, AlexNet has 8 layers - 5 Convolutional and 3
fully-connected. AlexNet just stacked a few more layers onto LeNet-5

First to implement Rectified Linear Units (RELUs) as activation functions
Also introduced Dropout
"""
deepplot.AlexNet()

in_ = Input(shape=(227, 227, 3), name='In')
c1 = Conv2D(96, 11, 4, padding='valid', activation='relu')(in_)
s2 = MaxPool2D(3, 2)(c1)
c3 = Conv2D(256, 5, 1, padding='same', activation='relu')(s2)
s4 = MaxPool2D(3, 2)(c3)
c5 = Conv2D(384, 3, 1, padding='same', activation='relu')(s4)
c6 = Conv2D(384, 3, 1, padding='same', activation='relu')(c5)
c7 = Conv2D(256, 3, 1, padding='same', activation='relu')(c6)
s8 = MaxPool2D(3, 2)(c7)
f_ = Flatten()(s8)
F9 = Dense(4096, activation='relu')(f_)
drop_out = tf.keras.layers.Dropout(0.5)(F9)
F10 = Dense(4096, activation='relu')(drop_out)
drop_out_ = tf.keras.layers.Dropout(0.5)(F10)
Out = Dense(1000, activation='softmax')(drop_out_)

model = Model(in_, Out)
model.summary()

"""
To reduce overfitting, the authors used two regularization techniques.
First, they applied dropout with a 50% dropout rate during training
to the outputs of layers F9 and F10. Second, they performed data
augmentation by randomly shifting the training images by various offsets,
flipping them horizontally, and changing the lighting conditions.
"""
