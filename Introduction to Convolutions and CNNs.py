import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# Convolution
input_image = tf.constant([
    [3, 3, 2, 1, 0],
    [0, 0, 1, 3, 1],
    [3, 1, 2, 2, 3],
    [2, 0, 0, 2, 2],
    [2, 0, 0, 0, 1]], dtype=tf.float32)

filter_matrix = tf.constant([
    [0, 1, 2],
    [2, 2, 0],
    [0, 1, 2]
    ], dtype=tf.float32)


#Reshape the image so that:
# - It has a depth of 1
# - It resembles a batch of one image
input_ = tf.expand_dims(input_image, -1)
filter_ = tf.expand_dims(filter_matrix, -1)

input_ = tf.expand_dims(input_, 0)
filter_ = tf.expand_dims(filter_, 3)

input_.shape, filter_.shape
# convolving the input and the filter
output = tf.nn.conv2d(
    input_, filter_,
    strides = [1, 1, 1, 1], padding='VALID'
)

# input = 5 (n)
# filter = 3 (f)
# stride = 1 (s)
# padding = 0 (p)
# output = [(n+2p-f)/s+1] X [(n+2p-f)/s+1]
# output = [(5+(2*0)-3)/1 + 1] = [2+1] = 3
output.shape
# tf.squeeze is the inverse of tf.expand_dims
# remove batch to get single image
output = tf.squeeze(output, axis=0)
# remove depth from image to remain with height and width
output = tf.squeeze(output, axis=-1)
print(output.numpy())

r = tf.constant([
    [156, 155, 156, 158, 158],
      [153, 154, 157, 159, 159],
      [149, 151, 155, 158, 159],
      [146, 146, 149, 153, 158],
      [145, 153, 153, 148, 158]
])
g = tf.constant([
    [167, 166, 167, 169, 169],
      [164, 165, 168, 170, 170],
      [160, 162, 166, 169, 170],
      [156, 156, 159, 163, 168],
      [155, 153, 153, 158, 168]
])
b = tf.constant([
    [163, 162, 163, 165, 165],
      [160, 161, 164, 166, 166],
      [156, 158, 162, 165, 166],
      [155, 155, 158, 162, 167],
      [154, 152, 152, 157, 167]
])

r_filter = tf.constant([
    [-1, -1, 1],
      [0, 1, -1],
      [0, 1, 1],
])
g_filter = tf.constant([
    [1, 0, 0],
    [1, -1, -1],
    [1, 0, -1]
])
b_filter = tf.constant([
    [0, 1, 1],
    [0, 1, 0],
    [1, -1, 1]
])

def convolve(inputs, filters):
    padin =  tf.constant([[1, 1], [1, 1]])
    outputs = []
    for (i, j) in zip(inputs, filters):
        i = tf.pad(i, padin,'CONSTANT')
        input_ = tf.expand_dims(i, -1)
        filter_ = tf.expand_dims(j, -1)
        input_ = tf.expand_dims(input_, 0)
        filter_ = tf.expand_dims(filter_, 3)
        output = tf.nn.conv2d(
        input_, filter_,
        strides = [1, 1, 1, 1], padding='VALID'
        )
        outputs.append(output)
    return tf.squeeze(tf.convert_to_tensor(np.concatenate(outputs)), axis=-1)

out_res = convolve([r,g,b], [r_filter, g_filter, b_filter])
bias = 1
print(tf.math.reduce_sum(out_res, axis=0) + bias)

# Padding
# - Valid padding - no padding

# - Same padding
# number of paddings on top, bottom , left, right
paddings = tf.constant([[1, 1], [1, 1]])
paddings.shape # n, 2 where n is rank of tensor
padded_image = tf.pad(input_image, paddings, 'CONSTANT')
padded_image

# For a vector
input_vector = tf.constant([1, 2, 4, 7])
paddings = tf.constant([[1, 4]])
tf.pad(input_vector, paddings, 'CONSTANT')

# Applying non-linearity
# ReLU
a = tf.range(-10, 10, delta=0.01)
plt.plot(tf.nn.relu(a))
plt.show()

plt.plot(tf.nn.leaky_relu(a, alpha=0.0))
plt.grid(True)
plt.show()


a = tf.cast(a, tf.float32)
plt.plot(tf.nn.elu(a))
plt.grid(True)
plt.show()

plt.plot(tf.nn.sigmoid(a))
plt.grid(True)
plt.show()

plt.plot(tf.nn.tanh(a))
plt.grid(True)
plt.show()

plt.plot(tf.nn.sigmoid(a), label="Sigmoid")
plt.plot(tf.nn.tanh(a), label="tanh")
plt.legend()
plt.grid(True)
plt.show()


out_prob = tf.nn.softmax(a)
tf.math.reduce_sum(out_prob)
tf.math.reduce_all([tf.where(i >= 0 and i <=1, True, False) for i in out_prob])

# Pooling
# - Max Pooling
input_ = tf.constant([
    [1, 1, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 2, 3, 4]
], dtype=tf.float16)

input_depth = tf.expand_dims(input_, axis=-1)
input_batch = tf.expand_dims(input_depth, axis=0)

out = tf.nn.pool(
input=input_batch,
window_shape=(2, 2),
pooling_type="AVG",
strides=(2, 2),
padding='VALID')

print(tf.squeeze(tf.squeeze(out, axis=0), axis=-1).numpy())



# FC layer
flattened = tf.constant([1,2,3,4,5,6,7], dtype=tf.float32)

fc_layer = tf.keras.Sequential([
    tf.keras.Input(shape=(7,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

fc_layer(tf.expand_dims(flattened, axis=0))
