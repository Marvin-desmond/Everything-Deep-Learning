"""
RESNEXT
"""

import tensorflow as tf

from tensorflow.keras.layers import Lambda, Conv2D, MaxPool2D, BatchNormalization, concatenate, add, GlobalAvgPool2D, Dense
from tensorflow.keras import Model, Input, Sequential
import deepplot

deepplot.ResNextVar()

class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,input_channels,output_channels,kernel_size,
                padding='valid', strides=(1, 1),groups=1,**kwargs):
        super(GroupConv2D, self).__init__()
        if not input_channels % groups == 0:
            raise ValueError("The input channels must be divisible by the no. of groups")
        if not output_channels % groups == 0:
            raise ValueError("The output channels must be divisible by the no. of groups")
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(
            Conv2D(
            filters=self.group_out_num,
            kernel_size = kernel_size,
            strides = strides,
            padding=padding)
            )

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](
            inputs[:, :, :, i * self.group_in_num: (i+1)*self.group_in_num]
            )
            feature_map_list.append(x_i)
        out = concatenate(feature_map_list, axis=-1)
        return out


class ResNext_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(ResNext_BottleNeck, self).__init__()
        self.conv1 = Conv2D(filters=filters,
                            kernel_size=(1, 1),
                            strides=1,
                            padding='same'
                            )
        self.bn1 = BatchNormalization()
        self.conv2 = GroupConv2D(
                            input_channels=filters,
                            output_channels=filters,
                            kernel_size=(3, 3),
                            strides=strides,
                            padding='same',
                            groups=groups)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(filters=2*filters,
                            kernel_size=(1, 1),
                            strides=1,
                            padding='same')
        self.bn3 = BatchNormalization()
        self.shortcut_conv = Conv2D(filters=2*filters,
                            kernel_size=(1, 1),
                            strides=strides,
                            padding='same')
        self.shortcut_bn = BatchNormalization()

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(add([x, shortcut]))
        return output

deepplot.ResNextArch()

def build_ResNext_block(filters, strides, groups, repeat_num):
    block = Sequential()
    block.add(ResNext_BottleNeck(filters=filters,
                                strides=strides,
                                groups=groups))
    for _ in range(1, repeat_num):
        block.add(ResNext_BottleNeck(filters=filters,
                                strides=1,
                                groups=groups))
    return block


NUM_CLASSES = 10

class ResNext(Model):
    def __init__(self, repeat_num_list, cardinality):
        if len(repeat_num_list) != 4:
            raise ValueError('The length of repeat_num_list must be 4')
        super(ResNext, self).__init__()
        self.conv1 = Conv2D(filters=64,
                            kernel_size=7,
                            strides=2,
                            padding='same')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size=3,
                            strides=2,
                            padding='same')
        self.block1 = build_ResNext_block(
                            filters=128,
                            strides=1,
                            groups=cardinality,
                            repeat_num = repeat_num_list[0])
        self.block2 = build_ResNext_block(
                            filters=256,
                            strides=2,
                            groups=cardinality,
                            repeat_num = repeat_num_list[1])
        self.block3 = build_ResNext_block(
                            filters=512,
                            strides=2,
                            groups=cardinality,
                            repeat_num = repeat_num_list[2])
        self.block4 = build_ResNext_block(
                            filters=1024,
                            strides=2,
                            groups=cardinality,
                            repeat_num = repeat_num_list[3])
        self.pool2 = GlobalAvgPool2D()
        self.fc = Dense(units=NUM_CLASSES, activation='softmax')
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)

        return x

def ResNext50():
    return ResNext(repeat_num_list=[3, 4, 6, 3],
                   cardinality=32)

def ResNext101():
    return ResNext(repeat_num_list=[3, 4, 23, 3],
                   cardinality=32)

resnext50 = ResNext50()
resnext101 = ResNext101()

dummy_in = tf.random.uniform((10, 224, 224, 3))

for model in [resnext50, resnext101]:
    out = model(dummy_in)
    print(model.count_params())
