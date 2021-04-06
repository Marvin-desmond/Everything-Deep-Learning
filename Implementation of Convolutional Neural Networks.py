import tensorflow as tf
import matplotlib.pyplot as plt

arch = "figures/CNN architecture.png"

img = plt.imread(arch)
plt.figure(figsize=(24, 20))
plt.imshow(img)
plt.axis('off')
plt.show()

from tensorflow.keras import Input, Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ZeroPadding2D, Flatten

model = Sequential([
 Input(shape=(227, 227, 3), name="Input"),
 # Layer 1

 Conv2D(filters=96, kernel_size=11, strides=4, activation="relu",name="Input"),
 MaxPooling2D(pool_size=3, strides=2),
 # Layer 2
 ZeroPadding2D(padding=2),
 Conv2D(filters=256, kernel_size=5, strides=1, activation="relu"),
 MaxPooling2D(pool_size=3, strides=2),
 # Layer 3
 ZeroPadding2D(padding=1),
 Conv2D(filters=384, kernel_size=3, strides=1, activation="relu"),
 # Layer 4
 ZeroPadding2D(padding=1),
 Conv2D(filters=384, kernel_size=3, strides=1, activation="relu"),
 # Layer 5
 ZeroPadding2D(padding=1),
 Conv2D(filters=256, kernel_size=3, strides=1, activation="relu"),

 MaxPooling2D(pool_size=3, strides=2),

 Flatten(),

 Dense(4096, activation="relu"),
 Dense(4096, activation="relu"),
 Dense(1000, activation="softmax")
], name="CNN model")

model.summary()

for layer in model.layers:
    print(layer.name, layer.input_shape, layer.output_shape, layer.count_params())

model.name, model.input_shape, model.output_shape, model.count_params()

"""
If you have 10 filters that are 3 x 3 x 3 in one layer of a neural network,
how many parameters does that layer have ?

Answer :

3 x 3 x 3 + bias = 28 => 1 filter
28 x 10 => 280 => 10 filters

Conv Layers : f_l x f_l x n_c (l-1) x n_c(l) + n_c(l)

Dense Layer : n_c(l) x n_c (l-1) + n_c(l)

n(l-1) + 2p(l) - f(l)/s(l) + 1
(227 + 0 - 11 / 4) + 1

"""
