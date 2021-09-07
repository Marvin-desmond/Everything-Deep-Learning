# MOBILENET

import tensorflow as tf
import deepplot

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Add, Lambda, GlobalAvgPool2D, Dense
ReLU = tf.keras.activations.relu

import matplotlib.pyplot as plt

deepplot.InvertedResidual()
def InvertedResidual(in_, expand=64, squeeze=16):
    x = Conv2D(expand, 1)(in_)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = DepthwiseConv2D(3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(squeeze, 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return Add()([in_, x])

deepplot.LBottleNeck()

"""
Linear BottleNeck
"""
class LinearBottleNeck(tf.keras.layers.Layer):
    def __init__(self, out=16, expansion_factor=6, strides=2):
        super(LinearBottleNeck, self).__init__()
        self.expansion_factor = expansion_factor
        self.out = out
        self.bn1 = BatchNormalization()
        ### relu6
        self.dwconv = DepthwiseConv2D(3, strides=strides, padding='same')
        self.bn2 = BatchNormalization()
        ### relu6
        self.conv3 = Conv2D(out, 1)
        # residual below
        self.res = Conv2D(out, kernel_size=1, strides=strides, padding='same')

    def build(self, input_shape):
        self.conv1 = Conv2D(self.expansion_factor*int(input_shape[-1]), 1)

    def call(self, in_):
        x = self.conv1(in_)
        x = self.bn1(x)
        x = tf.nn.relu6(x)
        ####
        x = self.dwconv(x)
        x = self.bn2(x)
        x = tf.nn.relu6(x)
        ####
        x = self.conv3(x)
        ####
        return Add()([x, self.res(in_)])


deepplot.MBnetArch()

architecture = [
# Input
{"input":[224, 3],"exp_rate":None,"out":32, "n":1, "s":2},
# Bottlenecks
{"input":[112, 32],"exp_rate":1,"out":16, "n":1, "s":1},
{"input":[112, 16],"exp_rate":6,"out":24, "n":2, "s":2},
{"input":[56, 24],"exp_rate":6,"out":32, "n":3, "s":2},
{"input":[28, 32],"exp_rate":6,"out":64, "n":4, "s":2},
{"input":[14, 64],"exp_rate":6,"out":96, "n":3, "s":1},
{"input":[14, 96],"exp_rate":6,"out":160, "n":3, "s":2},
{"input":[7, 160],"exp_rate":6,"out":320, "n":1, "s":1},
{"input":[7, 320],"exp_rate":None,"out":1280, "n":1, "s":1},
# AveragePool2D
{"input":[7, 1280],"exp_rate":None,"out":None, "n":1, "s":None},
# Output Conv
{"input":[1, 1280],"exp_rate":None,"out":1000, "n":None, "s":None},
]


layers = []
args = architecture[0]

for _ in range(args["n"]):
    layers.append(Conv2D(args["out"], kernel_size=3, strides=args["s"], padding="same"))

deepplot.MBnetArch()

bottlenecks = architecture[1:-3]

for bottleneck in bottlenecks:
    for i in range(bottleneck["n"]):
        if (bottleneck["s"]==2):
            strides = 2 if i == 0 else 1
        else:
            strides = 1
        layers.append(LinearBottleNeck(
                out=bottleneck['out'],
                expansion_factor=bottleneck["exp_rate"],
                strides=strides
                ))

conv_after_neck = architecture[-3]

for _ in range(conv_after_neck["n"]):
    layers.append(Conv2D(conv_after_neck["out"], kernel_size=1, strides=conv_after_neck["s"], padding="same"))

avgpool = architecture[-2]

for _ in range(avgpool["n"]):
    layers.append(GlobalAvgPool2D())

last_layer = architecture[-1]

layers.append(Dense(last_layer["out"]))

mobilenet = tf.keras.Sequential([*layers])

input_sample = tf.random.uniform((1, 224, 224, 3))
out = mobilenet(input_sample)

mobilenet.summary()
