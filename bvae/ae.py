'''
vae.py
contains the setup for autoencoders.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))

def test():
    import os
    import numpy as np
    from PIL import Image
    from tensorflow.python.keras.preprocessing.image import load_img
    from models import Darknet19Encoder, Darknet19Decoder

    inputShape = (256, 256, 3)
    batchSize = 10
    latentSize = 1000

    img = load_img(os.path.join('..','images', 'img.jpg'), target_size=inputShape[:-1])
    img.show()

    img = np.array(img, dtype=np.float32) / 255 - 0.5
    # img = np.expand_dims(img, axis=0)
    img = np.array([img]*batchSize)

    encoder = Darknet19Encoder(inputShape, batchSize, latentSize, 'bvae')
    decoder = Darknet19Decoder(inputShape, batchSize, latentSize)
    ae = AutoEncoder(encoder, decoder)

    ae.ae.compile(optimizer='adam', loss='mean_absolute_error')
    while True:
        ae.ae.fit(img, img,
                    epochs=100,
                    batch_size=batchSize)
        pred = ae.ae.predict(img)
        pred = np.uint8((pred + 0.5)* 255)

        pred = Image.fromarray(pred[0])
        pred.show()

if __name__ == "__main__":
    test()