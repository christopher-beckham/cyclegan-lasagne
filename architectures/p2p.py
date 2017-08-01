import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *
from lasagne.updates import *
from lasagne.objectives import *
from keras.preprocessing.image import ImageDataGenerator
import os
import sys
from layers import BilinearUpsample2DLayer

# custom layers

def _remove_trainable(layer):
    for key in layer.params:
        layer.params[key].remove('trainable')
        
def Convolution(layer, f, k=3, s=2, border_mode='same', **kwargs):
    return Conv2DLayer(layer, num_filters=f, filter_size=(k,k), stride=(s,s), pad=border_mode, nonlinearity=linear)

def Deconvolution(layer, f, k=2, s=2, **kwargs):
    return Deconv2DLayer(layer, num_filters=f, filter_size=(k,k), stride=(s,s), nonlinearity=linear)

def concatenate_layers(layers, **kwargs):
    return ConcatLayer(layers, axis=1)

def UpsampleBilinear(layer, f, s=2):
    layer = BilinearUpsample2DLayer(layer, s)
    layer = Convolution(layer, f, s=1)
    return layer



def g_unet_256(in_shp, is_a_grayscale, is_b_grayscale, nf=64, mul_factor=[1,2,4,8,8,8,8,8], act=tanh, dropout=0., bilinear_upsample=False, disable_skip=False):
    """
    The UNet in Costa's pix2pix implementation with some added arguments.
    is_a_grayscale:
    is_b_grayscale:
    nf: multiplier for # feature maps
    dropout: add 0.5 dropout to the first 3 conv-blocks in the decoder.
      This is based on the architecture used in the original pix2pix paper.
      No idea how it fares when combined with num_repeats...
    num_repeats:
    """
    assert len(mul_factor)==8
    if bilinear_upsample:
        ups = UpsampleBilinear
    else:
        ups = Deconvolution
    i = InputLayer((None, 1 if is_a_grayscale else 3, 256, 256))
    mf = mul_factor
    

    
    # 1,2,4,8,8,8,8,8

    mf = mul_factor
    
    # in_ch x 256 x 256
    conv1 = Convolution(i, nf*mf[0])
    conv1 = BatchNormLayer(conv1)
    x = NonlinearityLayer(conv1, nonlinearity=leaky_rectify)
    # nf x 128 x 128
    conv2 = Convolution(x, nf * mf[1])
    conv2 = BatchNormLayer(conv2)
    x = NonlinearityLayer(conv2, nonlinearity=leaky_rectify)
    # nf*2 x 64 x 64
    conv3 = Convolution(x, nf * mf[2])
    conv3 = BatchNormLayer(conv3)
    x = NonlinearityLayer(conv3, nonlinearity=leaky_rectify)
    # nf*4 x 32 x 32
    conv4 = Convolution(x, nf * mf[3])
    conv4 = BatchNormLayer(conv4)
    x = NonlinearityLayer(conv4, nonlinearity=leaky_rectify)
    # nf*8 x 16 x 16
    conv5 = Convolution(x, nf * mf[4])
    conv5 = BatchNormLayer(conv5)
    x = NonlinearityLayer(conv5, nonlinearity=leaky_rectify)
    # nf*8 x 8 x 8
    conv6 = Convolution(x, nf * mf[5])
    conv6 = BatchNormLayer(conv6)
    x = NonlinearityLayer(conv6, nonlinearity=leaky_rectify)
    # nf*8 x 4 x 4
    conv7 = Convolution(x, nf * mf[6])
    conv7 = BatchNormLayer(conv7)
    x = NonlinearityLayer(conv7, nonlinearity=leaky_rectify)
    # nf*8 x 2 x 2
    conv8 = Convolution(x, nf * mf[7], k=2, s=1, border_mode='valid')
    conv8 = BatchNormLayer(conv8)
    x = NonlinearityLayer(conv8, nonlinearity=leaky_rectify)
    
    #dconv1 = Deconvolution(x, nf * 8,
    #                       k=2, s=1)
    dconv1 = Deconvolution(x, nf * mf[6], k=2, s=1)
    dconv1 = BatchNormLayer(dconv1) #2x2
    x = concatenate_layers([dconv1, conv7])
    if disable_skip:
        x = dconv1
    x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
    # nf*(8 + 8) x 2 x 2
    dconv2 = ups(x, nf * mf[5])
    dconv2 = BatchNormLayer(dconv2)
    x = concatenate_layers([dconv2, DropoutLayer(conv6, dropout) if dropout>0 else conv6])
    if disable_skip:
        x = dconv2
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 4 x 4
    dconv3 = ups(x, nf * mf[4])
    dconv3 = BatchNormLayer(dconv3)
    x = concatenate_layers([dconv3, DropoutLayer(conv5, dropout) if dropout>0 else conv5])
    if disable_skip:
        x = dconv3
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 8 x 8
    dconv4 = ups(x, nf * mf[3])
    dconv4 = BatchNormLayer(dconv4)
    x = concatenate_layers([dconv4, DropoutLayer(conv4, dropout) if dropout>0 else conv4])
    if disable_skip:
        x = dconv4
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 16 x 16
    dconv5 = ups(x, nf * mf[2])
    dconv5 = BatchNormLayer(dconv5)
    x = concatenate_layers([dconv5, DropoutLayer(conv3, dropout) if dropout>0 else conv3])
    if disable_skip:
        x = dconv5
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 32 x 32
    dconv6 = ups(x, nf * mf[1])
    dconv6 = BatchNormLayer(dconv6)
    x = concatenate_layers([dconv6, DropoutLayer(conv2, dropout) if dropout>0 else conv2])
    if disable_skip:
        x = dconv6
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(4 + 4) x 64 x 64
    dconv7 = ups(x, nf * mf[0])
    dconv7 = BatchNormLayer(dconv7)
    x = concatenate_layers([dconv7, DropoutLayer(conv1, dropout) if dropout>0 else conv1])
    if disable_skip:
        x = dconv7
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(2 + 2) x 128 x 128
    dconv9 = ups(x, 1 if is_b_grayscale else 3)
    # out_ch x 256 x 256
    #act = 'sigmoid' if is_binary else 'tanh'
    out = NonlinearityLayer(dconv9, act)
    
    return out



def g_unet(in_shp, is_a_grayscale, is_b_grayscale, nf=64, dropout=False, num_repeats=0, bilinear_upsample=False):
    """
    The UNet in Costa's pix2pix implementation with some added arguments.
    is_a_grayscale:
    is_b_grayscale:
    nf: multiplier for # feature maps
    dropout: add 0.5 dropout to the first 3 conv-blocks in the decoder.
      This is based on the architecture used in the original pix2pix paper.
      No idea how it fares when combined with num_repeats...
    num_repeats:
    """
    #assert in_shp in [512]
    def padded_conv(nf, x):
        x = Convolution(x, nf,s=1,k=3)
        x = BatchNormLayer(x)
        x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
        return x
    i = InputLayer((None, 1 if is_a_grayscale else 3, in_shp, in_shp))
    # in_ch x 512 x 512
    conv1 = Convolution(i, nf)
    conv1 = BatchNormLayer(conv1)
    x = NonlinearityLayer(conv1, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf, x)    
    # nf x 256 x 256
    conv2 = Convolution(x, nf * 2)
    conv2 = BatchNormLayer(conv2)
    x = NonlinearityLayer(conv2, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*2, x)
    # nf*2 x 128 x 128
    conv3 = Convolution(x, nf * 4)
    conv3 = BatchNormLayer(conv3)
    x = NonlinearityLayer(conv3, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*4, x)
    # nf*4 x 64 x 64
    conv4 = Convolution(x, nf * 8)
    conv4 = BatchNormLayer(conv4)
    x = NonlinearityLayer(conv4, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 32 x 32
    conv5 = Convolution(x, nf * 8)
    conv5 = BatchNormLayer(conv5)
    x = NonlinearityLayer(conv5, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 16 x 16
    conv6 = Convolution(x, nf * 8)
    conv6 = BatchNormLayer(conv6)
    x = NonlinearityLayer(conv6, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 8 x 8
    conv7 = Convolution(x, nf * 8)
    conv7 = BatchNormLayer(conv7)
    x = NonlinearityLayer(conv7, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 4 x 4
    conv8 = Convolution(x, nf * 8)
    conv8 = BatchNormLayer(conv8)
    x = NonlinearityLayer(conv8, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 2 x 2
    conv9 = Convolution(x, nf * 8, k=2, s=1, border_mode='valid')
    conv9 = BatchNormLayer(conv9)
    x = NonlinearityLayer(conv9, nonlinearity=leaky_rectify)
    # nf*8 x 1 x 1  
    dconv1 = Deconvolution(x, nf * 8,
                           k=2, s=1)
    dconv1 = BatchNormLayer(dconv1)
    if dropout:
        dconv1 = DropoutLayer(dconv1, p=0.5)
    x = concatenate_layers([dconv1, conv8])
    x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
    # nf*(8 + 8) x 2 x 2
    if not bilinear_upsample:
        dconv2 = Deconvolution(x, nf * 8)
    else:
        dconv2 = BilinearUpsample2DLayer(x, 2)
        dconv2 = Convolution(dconv2, nf*8, s=1)
    dconv2 = BatchNormLayer(dconv2)
    if dropout:
        dconv2 = DropoutLayer(dconv2, p=0.5)
    x = concatenate_layers([dconv2, conv7])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 4 x 4
    if not bilinear_upsample:
        dconv3 = Deconvolution(x, nf * 8)
    else:
        dconv3 = BilinearUpsample2DLayer(x, 2)
        dconv3 = Convolution(dconv3, nf*8, s=1)
    dconv3 = BatchNormLayer(dconv3)
    if dropout:
        dconv3 = DropoutLayer(dconv3, p=0.5)
    x = concatenate_layers([dconv3, conv6])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 8 x 8
    if not bilinear_upsample:
        dconv4 = Deconvolution(x, nf * 8)
    else:
        dconv4 = BilinearUpsample2DLayer(x, 2)
        dconv4 = Convolution(dconv4, nf*8, s=1)
    dconv4 = BatchNormLayer(dconv4)
    x = concatenate_layers([dconv4, conv5])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 16 x 16
    if not bilinear_upsample:
        dconv5 = Deconvolution(x, nf * 8)
    else:
        dconv5 = BilinearUpsample2DLayer(x, 2)
        dconv5 = Convolution(dconv5, nf*8, s=1)        
    dconv5 = BatchNormLayer(dconv5)
    x = concatenate_layers([dconv5, conv4])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 32 x 32
    if not bilinear_upsample:
        dconv6 = Deconvolution(x, nf * 4)
    else:
        dconv6 = BilinearUpsample2DLayer(x, 2)
        dconv6 = Convolution(dconv6, nf*4, s=1)                
    dconv6 = BatchNormLayer(dconv6)
    x = concatenate_layers([dconv6, conv3])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(4 + 4) x 64 x 64
    if not bilinear_upsample:
        dconv7 = Deconvolution(x, nf * 2)
    else:
        dconv7 = BilinearUpsample2DLayer(x, 2)
        dconv7 = Convolution(dconv7, nf*2, s=1)
    dconv7 = BatchNormLayer(dconv7)
    x = concatenate_layers([dconv7, conv2])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(2 + 2) x 128 x 128
    if not bilinear_upsample:
        dconv8 = Deconvolution(x, nf)
    else:
        dconv8 = BilinearUpsample2DLayer(x, 2)
        dconv8 = Convolution(dconv8, nf, s=1)
    dconv8 = BatchNormLayer(dconv8)
    x = concatenate_layers([dconv8, conv1])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(1 + 1) x 256 x 256
    dconv9 = Deconvolution(x, 1 if is_b_grayscale else 3)
    # out_ch x 512 x 512
    act = sigmoid if is_b_grayscale else tanh
    out = NonlinearityLayer(dconv9, act)
    return out

def discriminator(in_shp, is_grayscale, nf=32, act=sigmoid, mul_factor=[1,2,4,8], num_repeats=0, bn=False):
    i = InputLayer((None, 1 if is_grayscale else 3, in_shp, in_shp))
    x = i
    for m in mul_factor:
        for r in range(num_repeats+1):
            x = Convolution(x, nf*m, s=2 if r == 0 else 1)
            x = NonlinearityLayer(x, leaky_rectify)
            if bn:
                x = BatchNormLayer(x)
    x = Convolution(x, 1)
    out = NonlinearityLayer(x, act)
    # 1 x 16 x 16
    return out

# for debugging

def fake_generator(in_shp, is_a_grayscale, is_b_grayscale, act=tanh):
    i = InputLayer((None, 1 if is_a_grayscale else 3, in_shp, in_shp))
    c = Convolution(i, f=1 if is_b_grayscale else 3, s=1)
    c = NonlinearityLayer(c, act)
    return c

def fake_discriminator(in_shp, is_grayscale):
    i = InputLayer((None, 1 if is_grayscale else 3, in_shp, in_shp))
    c = Convolution(i,1)
    return c

if __name__ == '__main__':
    l_out = g_unet(256, False, False)
    print l_out.output_shape
