import tensorflow as tf

scalar = tf.constant(4)
vector = tf.constant([1, 2, 3], dtype=tf.float32)
matrix = tf.constant([
[1, 2],
[3, 4]
])

scalar_var = tf.Variable(4)
vector_var = tf.Variable([1, 2, 3])
matrix_var = tf.Variable([
[1, 2],
[3, 4]
])

vector = tf.constant([1,2,3,4,5,6])
print(vector)
reshaped = tf.reshape(vector, -1)
print(reshaped)

vector
vector_numpy = vector.numpy()
print(vector_numpy)
type(vector_numpy)
vector_tensor = tf.convert_to_tensor(vector_numpy)
print(vector_tensor)

a = tf.constant([[1, 1], [1, 1]])
a = tf.ones((2, 2), dtype=tf.int32)
print(a)
b = tf.constant([[1, 2], [3, 4]])

print(tf.add(a,b))
print(a+b)
print(tf.subtract(a, b))
print(a-b)

print(tf.divide(a, b))
print(a/b)

print(tf.multiply(a, b))

print(a*b)
print(tf.matmul(a, b))

print(a@b)

ragged_tensor = tf.ragged.constant([
[1],
[1, 2, 3],
[2, 3]
])

string_tensor = tf.constant("The fox")


sparse_tensor = tf.sparse.SparseTensor(
indices= [[0,0], [1, 2]],
values=[3, 5],
dense_shape=[3, 3]
)

matrix

matrix.shape
matrix.ndim
tf.size(matrix)
tf.argmax(matrix)

# Perceptron
# MLP
# Architecture 
