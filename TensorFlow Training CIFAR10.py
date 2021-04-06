import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import CnnModel

import imp
imp.reload(CnnModel)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Train samples: {}".format(x_train.shape[0]))
print("Test samples: {}".format(x_test.shape[0]))

preprocessing = tf.keras.layers.experimental.preprocessing
# norm_layer.adapt(x_train[0])

import numpy as np 
mean = np.array([127.5] * 3)
norm_im = x_train[0]
norm_im = norm_layer(x_train[0])
norm_im = norm_im.numpy()
norm_im[:, :, 0].mean(), norm_im[:, :, 0].std() 
norm_im[:, :, 1].mean(), norm_im[:, :, 1].std() 
norm_im[:, :, 2].mean(), norm_im[:, :, 2].std() 

norm_im[:, :, 2].min(), norm_im[:, :, 2].max() 


# Option 1 : Using dataset in numpy format
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

class_names = ['airplane', 'automobile',
               'bird', 'cat', 'deer',
               'dog','frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.grid(False)
    plt.axis("off")
    plt.title(class_names[y_train[i].item()])
plt.show()


model = CnnModel.get_model(INPUT_SHAPE=(32, 32, 3), OUTPUT_CLASSES=10)

model.summary()

model.compile(
optimizer="Adam",
loss = 'sparse_categorical_crossentropy',
metrics=['accuracy']
)

model.fit(x_train, y_train, epochs = 1, batch_size=16, validation_split=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.batch(16)
tf.data.experimental.cardinality(train_dataset)

def create_dataset(data, labels, batch_size):
    def gen():
        for image, label in zip(data, labels):
            yield image, label 
    # ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((32, 32, 3), (1,)))
    ds = tf.data.Dataset.from_generator(gen, 
    output_signature=(
        tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)))
    return ds.batch(batch_size)

train_gen_data = create_dataset(x_test, y_test, 32)


model_ = tf.keras.Sequential([ 
    scaling,
    model,
])

model_.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

model_.fit(train_gen_data, epochs = 2, validation_data=)

for x, y in train_gen_data.take(1):
    print(x.shape, y.shape)

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import Sequential

# Doing preprocessing inside the model
# Standardization
# Data augmentation
augmentations = Sequential([
preprocessing.RandomFlip("horizontal"),
preprocessing.RandomRotation(0.1),
preprocessing.RandomZoom(0.1),
])

scaling = preprocessing.Rescaling(1.0 / 255)

model_ = Sequential([
augmentations,
scaling,
model
])

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    aug_img = model_(tf.expand_dims(x_train[i], 0))
    plt.imshow(aug_img[0, ...].numpy())
    plt.grid(False)
    plt.axis("off")
    plt.title(class_names[y_train[i].item()])
plt.show()

model_.compile(
optimizer="Adam",
loss = 'sparse_categorical_crossentropy',
metrics=['accuracy']
)

train_dataset = train_dataset.batch(16)

model_.fit(train_dataset, epochs = 1)




tensorflow.keras.layers.experimental.preprocessing
- Normalization
- RandomFlip("horizontal"),
- RandomRotation(0.1),
- RandomZoom(0.1),


- Normalize between 0 to 1
- preprocessing.Rescaling(1.0 / 255)
- Normalize between -1 to 1
- mean  = np.array([127.5] * 3)
- var = mean ** 2
- norm_layer = Normalization()
- norm_layer.set_weights([mean, var])
- mean 0 std 1
- norm_layer = Normalization()
- norm_layer.adapt(x_train[0])
