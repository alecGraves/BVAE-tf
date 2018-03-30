'''
models.py
contains models for use with the BVAE experiments.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, LeakyReLU, MaxPool2D, GlobalAveragePooling2D

class architecture(name_scope):
    '''
    generic architecture template
    '''
    def __init__(self, input_shape, name='net'):
        '''
        
        '''
        self.input_shape = input_shape
        self.name = name

        self.net = None
        self.input_layer = None
        self.output_layer = None

    def build(self, input_shape):
        NotImplementedError('architecture must implement build function')


class darknet_19_encoder(object):
    '''
    a simple, fully convolutional architecture inspried by pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, input_shape=(None, None, None, None), name='darknet19_encoder')
        super().__init__(input_shape, name)
        self.input_layer, self.output_layer = self.build()

    def build(self):
        # create the input layer for feeding the netowrk
        in_layer = InputLayer(self.input_shape(:3), self.)

class darknet_53_encoder(object):
    '''
    a larger, fully convolutional architecture inspried by pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    https://pjreddie.com/media/files/papers/YOLOv3.pdf
    '''
    def __init__(self, input_shape=(None, None, None, None), name='darkent53_encoder'):
        '''
        input shape for the network, a name for the scope, and a data format.
        '''
        super().__init__(input_shape, name)
        self.input_layer, self.output_layer = self.build()

    def build():
        '''
        builds darknet53 encoder network
        '''
        pass

    def conv_block(self):
        '''
        adds a darknet conv block to the net
        '''
        pass

        

