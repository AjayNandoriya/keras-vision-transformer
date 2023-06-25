import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
if __package__ is None or __package__ == '':
    import sys
    PACKAGE_PATH = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(PACKAGE_PATH)

from keras_vision_transformer.models import create_swinunet_gen_model, create_conv_unet, create_conv_style_unet
from keras_vision_transformer.data_generator import DGS2S, DGStyle

def factor():
    s = 23
    k =8
    s2 = 17 + 2*28*k + k**2
    print(s2, np.sqrt(s2))

    a = 1 + 16 + 2*4*1 + 2*4*k + 2*1*k + k**2 + 2*23*k - 2*4*1

    a = (k+5)**2
    print(a)
    pass

def train(epochs:int=10):
    filepath = os.path.join(os.path.dirname(__file__), 'datasets', 'M2S')
    if 0:
        out_model_fname = os.path.join(filepath,'conv_model.h5')
        model = create_conv_unet()
        dg = DGS2S(filepath=filepath)

    if 1:
        out_model_fname = os.path.join(filepath,'conv_style_model.h5')
        model = create_conv_style_unet()
        dg = DGStyle(filepath=filepath)

    # model = create_swinunet_gen_model()

    
    
    if os.path.isfile(out_model_fname):
        model.load_weights(out_model_fname)

    if epochs>0:
        model.fit(dg, epochs=epochs)
        model.save(out_model_fname)

    for i in range(len(dg)):
        X,Y  = dg.__getitem__(i)
        out_img = model.predict(X)
        fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
        axes[0].imshow(X[0,:,:,0])
        axes[1].imshow(out_img[0,:,:,0])
        axes[2].imshow(Y[0,:,:,0])
        plt.show()
        pass

if __name__ == '__main__':
    # factor()
    train(epochs=10)
    pass