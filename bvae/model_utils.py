'''
model_utils.py
contains custom blocks, etc. for building mdoels.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose, 
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, 
            Reshape, GlobalAveragePooling2D, Layer)

class ConvBnLRelu(object):
    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
    # return conv + bn + leaky_relu model
    def __call__(self, net, training=None):
        net = Conv2D(self.filters, self.kernelSize, strides=self.strides, padding='same')(net)
        net = BatchNormalization()(net, training=training)
        net = LeakyReLU()(net)
        return net

