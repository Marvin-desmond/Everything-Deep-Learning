import tensorflow as tf

from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Dense, ZeroPadding2D, Flatten, InputLayer

"""
There are three ways to create models in TensorFlow / Keras:
1. The Sequential model, which is very straightforward
(a simple list of layers), but is limited to single-input,
single-output stacks of layers (as the name gives away).
"""
model = Sequential([
 Input(shape=(227, 227, 3), name="Input"),
 Conv2D(filters=96, kernel_size=11, strides=4, activation="relu"),
 MaxPooling2D(pool_size=3, strides=2),
 ZeroPadding2D(padding=2),
 Conv2D(filters=256, kernel_size=5, strides=1, activation="relu"),
 MaxPooling2D(pool_size=3, strides=2),
 ZeroPadding2D(padding=1),
 Conv2D(filters=384, kernel_size=3, strides=1, activation="relu"),
 ZeroPadding2D(padding=1),
 Conv2D(filters=384, kernel_size=3, strides=1, activation="relu"),
 ZeroPadding2D(padding=1),
 Conv2D(filters=256, kernel_size=3, strides=1, activation="relu"),
 MaxPooling2D(pool_size=3, strides=2),
 Flatten(),
 Dense(4096, activation="relu"),
 Dense(4096, activation="relu"),
 Dense(1000, activation="softmax")
], name="CNNmodel")

model.summary()

"""
2. The Functional API, which is an easy-to-use, fully-featured API
 that supports arbitrary model architectures. For most people and
 most use cases, this is what you should be using. This is the Keras
 "industry strength" model.

 This API create models that are more flexible than the
 tf.keras.Sequential API. The functional API can handle
 models with non-linear topology, shared layers, and
 even multiple inputs or outputs.
"""

inputs = Input(shape=(227, 227, 3))
x = Conv2D(filters=96, kernel_size=11, strides = 4, activation="relu")(inputs)
x = MaxPooling2D(pool_size=3, strides=2)(x)
x = ZeroPadding2D(padding=2)(x)
x = Conv2D(filters=256, kernel_size=5, strides = 1, activation="relu")(x)
x = MaxPooling2D(pool_size=3, strides=2)(x)
x = ZeroPadding2D(padding=1)(x)
x = Conv2D(filters=384, kernel_size=3, strides = 1, activation="relu")(x)
x = ZeroPadding2D(padding=1)(x)
x = Conv2D(filters=384, kernel_size=3, strides = 1, activation="relu")(x)
x = ZeroPadding2D(padding=1)(x)
x = Conv2D(filters=256, kernel_size=3, strides = 1, activation="relu")(x)
x = MaxPooling2D(pool_size=3, strides=2)(x)
x = Flatten()(x)
x = Dense(4096)(x)
x = Dense(4096)(x)
outputs = Dense(1000)(x)

model_ = Model(inputs=inputs, outputs=outputs, name="CNNmodel")

model_.summary()


"""
3. Model subclassing is where you implement everything from scratch on
 your own. Use this if you have complex, out-of-the-box research use cases.
"""

class triBlockArchitecture(tf.keras.layers.Layer):
    def __init__(self, block=[True, True, True], f=1, k=1, p=1, s=1):
        self.block = block
        super(triBlockArchitecture, self).__init__()
        self.pad = ZeroPadding2D(p)
        self.conv = Conv2D(filters=f, kernel_size=k, strides=s, activation="relu")
        self.maxpool = MaxPooling2D(pool_size=3, strides=2)

    def call(self, x):
        if self.block[0]:
            x = self.pad(x)
        if self.block[1]:
            x = self.conv(x)
        if self.block[2]:
            x = self.maxpool(x)
        return x


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class CnnModel(tf.keras.Model):
    def __init__(self, in_shape, output_classes):
        super(CnnModel, self).__init__()
        self.layer1 = triBlockArchitecture([False, True, True], f=96, k=11, s=4)
        self.layer2 = triBlockArchitecture(f=256, k=5, p=2, s=1)
        self.layer3 = triBlockArchitecture([True, True, False], f=384, k=3, s=1)
        self.layer3_1 = triBlockArchitecture([True, True, False], f=384, k=3, s=1)
        self.layer4 = triBlockArchitecture(f=256, k=3, s=1)
        self.linear = Linear(units=4096)
        self.linear_1 = Linear(units=4096)
        self.classifier = Linear(units=1000)
        self.flatten = Flatten()

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_1(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.linear_1(x)
        return self.classifier(x)

    def model(self):
        x = Input(shape=(227, 227, 3))
        return Model(inputs=[x], outputs=self.call(x))

model_sub = CnnModel((227, 227, 3), 196)
model_sub.model().summary()
