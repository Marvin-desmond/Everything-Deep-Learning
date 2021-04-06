# import TF library
import tensorflow as tf
import deepplot
# Perceptron
"""
The Perceptron is one of the simplest ANN architectures, invented in 1957
by Frank Rosenblatt.
It is based on a slightly different artificial neuron called a
threshold logic unit (TLU), or sometimes a linear threshold unit (LTU).
"""
# Plot Threshold Logic Unit
deepplot.TLU()
"""
A perceptron is a linear, binary classifier. It finds the most articulate
boundary between two classes. To instantiate a more intuitive description,
a perceptron is able to perceive the difference between two given classes
and detect what class a given data point belongs to.
It is able to perceive representative space that separates one data point
from another, linearly.
"""
# Plot Perceptron
deepplot.Perceptron()
# Implement the model for Perceptron

"""
A model, abstractly, is
A function that computes something on tensors (a forward pass)
Some variables that can be updated in response to training

Most models are made of layers.
Layers are functions with a known mathematical structure
that can be reused and have trainable variables.
"""
class Perceptron(tf.keras.Model):
    def __init__(self, in_features=1, out_features=1, name="Perceptron"):
        super(Perceptron, self).__init__()
        self.w = tf.Variable(
        tf.random.normal([in_features, out_features]),
        name="weights", trainable=True
        )
        self.b = tf.Variable(
        tf.zeros([out_features]),
        name='bias', trainable=True
        )
    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return y


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train)
x_train = tf.reshape(x_train, [60000,-1]) / 255.0
x_train.shape
y_train.shape

perceptron = Perceptron(in_features=784, out_features=10)
perceptron.compile(
optimizer='adam',
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy']
)
perceptron.fit(x_train, y_train, epochs=2, validation_split=0.2)
x_test_sample = tf.convert_to_tensor(x_test[0], dtype=tf.float32)
x_test_sample = tf.reshape(x_test_sample, -1)
x_test_sample = x_test_sample / 255.0
x_test_sample = tf.expand_dims(x_test_sample, axis=0)
x_test_sample.shape
prediction = tf.nn.softmax(perceptron(x_test_sample))
tf.argmax(tf.reshape(prediction, -1))
y_test[0]

# MLP
"""
An MLP is composed of one (passthrough) input layer, one or more layers of TLUs,
called hidden layers, and one final layer of TLUs called the output layer
The layers close to the input layer are usually called the lower layers, and
the ones close to the outputs are usually called the upper layers. Every layer
except the output layer includes a bias neuron and is fully connected to the
 next layer.
"""
# Implement MLP
deepplot.MLP()

class MLP(tf.keras.Model):
    def __init__(self, name="MLP"):
        super(MLP, self).__init__()
        self.layer_one = Perceptron(in_features=784, out_features=16)
        self.layer_two = Perceptron(in_features=16, out_features=8)
        self.out_layer = Perceptron(in_features=8, out_features=10)
    def call(self, x):
        x = tf.nn.relu(self.layer_one(x))
        x = tf.nn.relu(self.layer_two(x))
        return self.out_layer(x)

mlp = MLP()

mlp.compile(
optimizer='adam',
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy']
)


mlp.fit(x_train, y_train, epochs=2, validation_split=0.2)

perceptron_model = tf.keras.Sequential([
tf.keras.Input(shape=(784,)),
tf.keras.layers.Dense(10),
])

perceptron_model.compile(
optimizer='adam',
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy']
)

perceptron_model.fit(x_train, y_train, epochs=2, validation_split=0.2)

mlp = tf.keras.Sequential([
tf.keras.Input(shape=(784,)),
tf.keras.layers.Dense(16, activation='relu'),
tf.keras.layers.Dense(32, activation='relu'),
tf.keras.layers.Dense(10)
])

mlp.compile(
optimizer='adam',
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy']
)

mlp.fit(x_train, y_train, epochs=2, validation_split=0.2)
