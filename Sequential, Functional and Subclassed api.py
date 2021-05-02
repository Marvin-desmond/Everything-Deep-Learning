import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Dense, ZeroPadding2D, Flatten, InputLayer

arch = "figures/CNN architecture.png"

img = plt.imread(arch)
plt.figure(figsize=(24, 20))
plt.imshow(img)
plt.axis('off')
plt.show()

"""
There are three ways to create models in TensorFlow / Keras:
1. The Sequential model
2. Functional api
3. Subclass api
"""

"""
1. THE SEQUENTIAL API
Appropriate for a plain stack of layers where each layer has
exactly one input tensor and one output tensor.
You can create a Sequential model by passing a list of layers
to the Sequential constructor.
"""

model = Sequential([
 Input(shape=(227, 227, 3), name="Input"),
 Conv2D(filters=96, kernel_size=11, strides=4, activation="relu", name="first_conv"),
 MaxPooling2D(pool_size=3, strides=2, name="first_max_pooling"),
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
 Dense(1000, activation="softmax")],
 name="CNNmodel")


# Once a model is "built", you can call its summary() method to display its contents
model.summary()


# You can also create a Sequential model incrementally via the add() method:
model_ = Sequential()
model_.add(Input(shape=(227, 227, 3)))
model_.add(Conv2D(64, 2, 2))
model_.add(MaxPooling2D(2, 2))
model_.summary()


"""
2. FUNCTIONAL API
For most people and most use cases, this is what you should be using.
This is the Keras "industry strength" model.

This API create models that are more flexible than the
tf.keras.Sequential API. The functional API can handle
models with non-linear topology, shared layers, and
even multiple inputs or outputs.

The main idea is that a deep learning model is usually a directed
acyclic graph (DAG) of layers. So the functional API is a way to
build graphs of layers.

It assumes a model as a basic graph with layers. So to build the graph,
you build the nodes, one after the other, while consider the layout
of our graph

For instance, in the model below, we start by creating the input
node.
Then, you create a new node in the graph of layers by calling a
layer on this inputs object.
With this in mind, more layers are added to the graph of layers until
the graph is complete.
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

# At this point, you can create a Model by specifying its inputs and
# outputs in the graph of layers
model_functional = Model(inputs=inputs, outputs=outputs, name="CNNmodel")

model_functional.summary()

assert model_functional.count_params() == model.count_params()

"""
3. SUBCLASS API
Where you implement everything from scratch on your own.
Use this if you have complex, out-of-the-box research use cases.
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
    def __init__(self, output_classes):
        super(CnnModel, self).__init__()
        self.layer1 = triBlockArchitecture([False, True, True], f=96, k=11, s=4)
        self.layer2 = triBlockArchitecture(f=256, k=5, p=2, s=1)
        self.layer3 = triBlockArchitecture([True, True, False], f=384, k=3, s=1)
        self.layer3_1 = triBlockArchitecture([True, True, False], f=384, k=3, s=1)
        self.layer4 = triBlockArchitecture(f=256, k=3, s=1)
        self.linear = Linear(units=4096)
        self.linear_1 = Linear(units=4096)
        self.classifier = Linear(units=output_classes)
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

    # def model(self):
    #     x = Input(shape=(227, 227, 3))
    #     return Model(inputs=[x], outputs=self.call(x))

model_sub = CnnModel(1000)
# model_sub.model().summary()
