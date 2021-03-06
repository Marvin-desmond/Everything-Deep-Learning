# Importing libraries
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Normalization, RandomFlip, RandomZoom, RandomRotation, Rescaling
import matplotlib.pyplot as plt
import CnnModel

# import imp
# imp.reload(CnnModel)

# Importing CIFAR-10 dataset
"""
The CIFAR-10 (Canadian Research For Advanced Research) is
a collection of images that are commonly used to train
machine learning and computer vision algorithms.
It is one of the most commonly used for machine learning
research.
The CIFAR-10 dataset contains 60_000 32 x 32(low-resolution)
color images in 10 different classes.
The 10 different classes represent airplanes, cars, birds,
cats, deer, dogs, frogs, horses, ships and trucks.
There are 6000 images in each class
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Sampling a subset for this tutorial
x_train, y_train, x_test, y_test = x_train[:5_000], y_train[:5_000], x_test[:2_000], y_test[:2_000]

class_names = ['airplane', 'automobile',
               'bird', 'cat', 'deer',
               'dog','frog', 'horse', 'ship', 'truck']

# Vizualization of a sample of images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.grid(False)
    plt.axis("off")
    plt.title(class_names[y_train[i].item()])
plt.show()

# Image normalization before model training
"""
It ensures that each input parameter (pixel, in this case)
has a similar data ditribution. This makes convergence
faster while training the network.

Normalizing images
1. Normalize pixel values between 0 and 1
2. Normalize pixel values between -1 and 1
3. Normalize mages with mean 0  and std 1
"""
# Normalize before modelling
# Range [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


# Model initialization
model = CnnModel.get_model(INPUT_SHAPE=(32, 32, 3), OUTPUT_CLASSES=10)

model.summary()

# Model compiling
model.compile(
optimizer="Adam",
loss = 'sparse_categorical_crossentropy',
metrics=['accuracy']
)

# Model training
model.fit(x_train, y_train, epochs = 1,
validation_split=0.1, batch_size=16)
"""
In general, it's a good practice to develop models that
take raw data as input, as opposed to models that take
already-preprocessed data. The reason being that, if your
model expects preprocessed data, any time you export
your model to use it elsewhere (in a web browser, in a
mobile app), you'll need to reimplement the same exact
preprocessing pipeline. This can be a bit tricky to do.
"""
# normalize in range [0, 1]
scaling_layer = Rescaling(1.0 / 255)
# normalize in range [-1, 1]
input_ = tf.keras.Input(shape=(32, 32, 3))
norm_neg_one_to_one = Normalization()
x = norm_neg_one_to_one(input_)
import numpy as np
mean = [127.5]*3
var = mean ** 2
norm_neg_one_to_one.set_weights([mean, var])
norm_neg_one_to_one.get_weights()

# normalize with mean 0 and std 1
norm_mean_std = Normalization()
norm_mean_std.adapt(x_train[0])

model_ = Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    norm_mean_std,
    model
])

model_.compile(
optimizer="Adam",
loss="sparse_categorical_crossentropy",
metrics=['accuracy'],
)
model_.fit(x_train, y_train, epochs=1, batch_size=16)
"""
When you don't have a large image dataset, it's a good practice
to artificially introduce sample diversity by applying random yet
realistic transformations to the training images. This helps
expose the model to different aspects of the training data while
slowing down overfitting

"""
augmentations = Sequential([
RandomFlip("horizontal"),
RandomRotation(0.1),
RandomZoom(0.1),
])

model_ = Sequential([
    augmentations,
    scaling_layer,
    model
])

model_.compile(
optimizer="Adam",
loss="sparse_categorical_crossentropy",
metrics=['accuracy'],
)

model_.fit(x_train, y_train, epochs=1, batch_size=16)
"""
WHY tf.data.Dataset OVER NUMPY ARRAY DATA

Dataset API is more efficient, because the data flows
directly to the device, bypassing the client.
Dataset API also improves training speed and is able
to handle large training sets, especially when you want
to deal with distributed computations.
"""

"""
Using tf.data.Dataset.from_tensor_slices adds the
whole dataset to the computational graph
"""
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(16)

model_.fit(train_dataset, epochs=1, validation_data=test_dataset)

"""
Using tf.data.Dataset.from_generator adds the dataset in batches
to the computational graph
"""

def create_dataset(data, labels, batch_size):
    def gen():
        for image, label in zip(data, labels):
            yield image, label
    # Deprecated
    # ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((32, 32, 3), (1,)))
    ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)))
    return ds.batch(batch_size)

train_gen_data = create_dataset(x_train, y_train, 32)
test_gen_data = create_dataset(x_test, y_test, 32)

model.fit(train_gen_data, epochs = 2, validation_data=test_gen_data)

model.evaluate(test_dataset)
model.evaluate(x_test, y_test)

sample_img = x_test[0]
im = tf.expand_dims(sample_img, 0)
predictions = model.predict(im)
index_of_prediction = tf.math.argmax(tf.reshape(prediction, -1))

print(f"Predicted: {class_names[index_of_prediction]}")
