'''
sample_layer.py
contains keras SampleLayer for bvae

created by shadySource

THE UNLICENSE
'''

from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class SampleLayer(Layer):
    '''
    Keras Layer to grab a random sample from a distribution (by multiplication)
    Computes "(normal)*logvar + mean" for the vae sampling operation
    (written for tf backend)

    Additionally,
        Applies regularization to the latent space representation.
        Can perform standard regularization or B-VAE regularization.

    call:
        pass in mean then logvar layers to sample from the distribution
        ex.
            sample = SampleLayer('bvae', 16)([mean, logvar])
    '''
    def __init__(self, latent_regularizer='bvae', beta=100., **kwargs):
        '''
        args:
        ------
        latent_regularizer : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer,
            (Unused if 'bvae' not selected)
        ------
        ex.
            sample = SampleLayer('bvae', 16)([mean, logvar])
        '''
        if latent_regularizer.lower() in ['bvae', 'vae']:
            self.reg = latent_regularizer
        else:
            self.reg = None
    
        if self.reg == 'bvae':
            self.beta = beta
        elif self.reg == 'vae':
            self.beta = 1.

        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # save the shape for distribution sampling
        super(SampleLayer, self).build(input_shape) # needed for layers

    def call(self, x, training=None):
        if len(x) != 2:
            raise Exception('input layers must be a list: mean and logvar')
        if len(x[0].shape) != 2 or len(x[1].shape) != 2:
            raise Exception('input shape is not a vector [batchSize, latentSize]')

        mean = x[0]
        logvar = x[1]

        # trick to allow setting batch at train/eval time
        if mean.shape[0].value == None or  logvar.shape[0].value == None:
            return mean + 0*logvar # Keras needs the *0 so the gradinent is not None

        if self.reg is not None:
            # kl divergence:
            latent_loss = -0.5 * (1 + logvar
                                - K.square(mean)
                                - K.exp(logvar))
            latent_loss = K.sum(latent_loss, axis=-1) # sum over latent dimension
            latent_loss = K.mean(latent_loss, axis=0) # avg over batch

            # use beta to force less usage of vector space:
            latent_loss = self.beta * latent_loss
            self.add_loss(latent_loss, x)

        def reparameterization_trick():
            epsilon = K.random_normal(shape=logvar.shape,
                              mean=0., logvar=1.)
            stddev = K.exp(logvar*0.5)
            return mean + stddev * epsilon * inf

        return K.in_train_phase(reparameterization_trick, mean + 0*logvar, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape[0]