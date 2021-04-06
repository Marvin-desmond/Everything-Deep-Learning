import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ZeroPadding2D, Flatten

def get_model(INPUT_SHAPE=(227, 227, 3), OUTPUT_CLASSES=196):
    model = Sequential([
     Input(shape=INPUT_SHAPE, name="Input"),
     # Layer 1
     Conv2D(filters=16, kernel_size=3, strides=4, activation="relu", padding="same"),
     MaxPooling2D(pool_size=3, strides=2),
     # Layer 3
     ZeroPadding2D(padding=1),
     Conv2D(filters=32, kernel_size=3, strides=1, activation="relu"),
     # Layer 4
     ZeroPadding2D(padding=1),
     Conv2D(filters=32, kernel_size=3, strides=1, activation="relu"),
     # Layer 5
     ZeroPadding2D(padding=1),
     Conv2D(filters=16, kernel_size=3, strides=1, activation="relu"),
     MaxPooling2D(pool_size=3, strides=2),
     Flatten(),
     Dense(4096, activation="relu"),
     Dense(4096, activation="relu"),
     Dense(OUTPUT_CLASSES, activation="softmax")
    ])
    return model
