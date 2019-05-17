'''
models.py
contains models for use with the BVAE experiments.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
            Reshape, GlobalAveragePooling2D)
from tensorflow.python.keras.models import Model

from model_utils import ConvBnLRelu
from sample_layer import SampleLayer

class Architecture(object):
    '''
    generic architecture template
    '''
    def __init__(self, inputShape=None, batchSize=None, latentSize=None):
        '''
        params:
        ---------
        inputShape : tuple
            the shape of the input, expecting 3-dim images (h, w, 3)
        batchSize : int
            the number of samples in a batch
        latentSize : int
            the number of dimensions in the two output distribution vectors -
            mean and std-deviation
        latentSize : Bool or None
            True forces resampling, False forces no resampling, None chooses based on K.learning_phase()
        '''
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.latentSize = latentSize

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class Darknet19Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output

    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(256, 256, 3), batchSize=None,
                 latentSize=1000, latentConstraints='bvae', beta=100., training=None):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.training=training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
        net = ConvBnLRelu(32, kernelSize=3)(inLayer, training=self.training) # 1
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training) # 2
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 3
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training) # 4
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 5
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 6
        net = ConvBnLRelu(128, kernelSize=1)(net, training=self.training) # 7
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 8
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 9
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training) # 10
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 11
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training) # 12
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 13
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 14
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 15
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 16
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 17
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 18

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        logvar = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        logvar = GlobalAveragePooling2D()(logvar)

        sample = SampleLayer(self.latentConstraints, self.beta)([mean, logvar], training=self.training)

        return Model(inputs=inLayer, outputs=sample)

class Darknet19Decoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=None, latentSize=1000, training=None):
        self.training=training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)

        # TODO try inverting num filter arangement (e.g. 512, 1204, 512, 1024, 512)
        # and also try (1, 3, 1, 3, 1) for the filter shape
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(128, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(32, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training)

        # net = ConvBnLRelu(3, kernelSize=1)(net, training=self.training)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same', activation="tanh")(net)

        return Model(inLayer, net)

class Darknet53Encoder(Architecture):
    '''
    a larger, fully convolutional architecture inspried by
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    https://pjreddie.com/media/files/papers/YOLOv3.pdf
    '''
    def __init__(self, inputShape=(None, None, None, None), name='darkent53_encoder'):
        '''
        input shape for the network, a name for the scope, and a data format.
        '''
        super().__init__(inputShape, name)
        self.Build()

    def Build(self):
        '''
        builds darknet53 encoder network
        '''
        raise NotImplementedError('this architecture is not complete')

    def ConvBlock(self):
        '''
        adds a darknet conv block to the net
        '''
        raise NotImplementedError('this architecture is not complete')


def test():
    d19e = Darknet19Encoder()
    d19e.model.summary()
    d19d = Darknet19Decoder()
    d19d.model.summary()

if __name__ == '__main__':
    test()
