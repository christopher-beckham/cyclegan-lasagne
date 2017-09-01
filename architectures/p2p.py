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
sys.path.append("..")
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

def resblock(layer, nf, s=1, norm_layer=BatchNormLayer, decode=False):
    left = layer
    if not decode:
        left = Convolution(left, f=nf, s=s)
    else:
        # upsample using bilinear sampling using the
        # custom stride and have the conv's s=1
        if s > 1:
            left = BilinearUpsample2DLayer(left, s)
        left = Convolution(left, f=nf, s=1)
    left = norm_layer(left)
    left = NonlinearityLayer(left, leaky_rectify)
    left = Convolution(left, f=nf, s=1) # shape-preserving, always
    left = norm_layer(left)
    # traditionally, i padded feature maps,
    # but here, we learn a projection
    right_ds = layer
    if not decode:
        right_ds = Convolution(right_ds, k=1, f=nf, s=s)
        right_ds = BatchNormLayer(right_ds)
    else:
        # upsample using bilinear sampling,
        # then do the 1x1 convolution to match dims
        # (don't stride the 1x1 conv, we already did
        # that with the bilinear upsample)
        raise Exception("...")
        right_ds = BilinearUpsample2DLayer(right_ds, s)
        right_ds = Convolution(right_ds, k=1, f=nf, s=1)
        right_ds = norm_layer(right_ds)
    add = ElemwiseSumLayer([left, right_ds])
    add = NonlinearityLayer(add, leaky_rectify)
    return add

def conv_bn_relu(layer, nf, s=1, norm_layer=BatchNormLayer):
    conv = layer
    conv = Convolution(conv, nf, s=s)
    conv = norm_layer(conv)
    conv = NonlinearityLayer(conv, nonlinearity=leaky_rectify)
    return conv

def up_conv_bn_relu(layer, nf, norm_layer=BatchNormLayer):
    conv = layer
    conv = Deconvolution(conv, nf)
    conv = norm_layer(conv)
    conv = NonlinearityLayer(conv, nonlinearity=leaky_rectify)
    return conv

def block9(in_shp, is_a_grayscale, is_b_grayscale, nf=64, instance_norm=False):
    norm_layer = BatchNormLayer if not instance_norm else InstanceNormLayer
    i = InputLayer((None, 1 if is_a_grayscale else 3, in_shp, in_shp))
    conv = i
    conv = norm_layer(Conv2DLayer(conv, num_filters=32, filter_size=7, pad='same', nonlinearity=leaky_rectify)) # c7s1
    conv = conv_bn_relu(conv, nf=nf, s=2, norm_layer=norm_layer) # d64
    conv = conv_bn_relu(conv, nf=nf*2, s=2, norm_layer=norm_layer) # d128
    for r in range(9):
        conv = resblock(conv, nf=nf*4, s=1, norm_layer=norm_layer) # R 128
    conv = up_conv_bn_relu(conv, nf=nf*2, norm_layer=norm_layer) # u64
    conv = up_conv_bn_relu(conv, nf=nf, norm_layer=norm_layer) # u32
    conv = Conv2DLayer(conv, num_filters=1 if is_b_grayscale else 3, filter_size=7, pad='same',
                       nonlinearity=sigmoid if is_b_grayscale else tanh) # c7s1
    return conv


def net_256_2_resblock(in_shp, is_a_grayscale, is_b_grayscale, nf=64, num_repeats=0, act=tanh):
    i = InputLayer((None, 1 if is_a_grayscale else 3, in_shp, in_shp))
    # 1,2,4,8,8,8,8,8
    mf = [1,2,4]
    enc = resblock(i, nf*mf[0], s=2) # 128
    for r in range(num_repeats):
        enc = resblock(enc, nf*mf[0], s=1)
    #
    enc = resblock(enc, nf*mf[1], s=2) # 64
    for r in range(num_repeats):
        enc = resblock(enc, nf*mf[1], s=1)
    #
    enc = resblock(enc, nf*mf[2], s=2) # 32
    for r in range(num_repeats):
        enc = resblock(enc, nf*mf[2], s=1)
    x = enc
    dec = x
    # decode
    dec = resblock(dec, nf*mf[1], s=2, decode=True) # 64
    for r in range(num_repeats):
        dec = resblock(dec, nf*mf[1], s=1, decode=True)
    #
    dec = resblock(dec, nf*mf[0], s=2, decode=True) # 128
    for r in range(num_repeats):
        dec = resblock(dec, nf*mf[0], s=1, decode=True)
    #
    dec = UpsampleBilinear(dec, 1 if is_b_grayscale else 3) # 256
    dec = NonlinearityLayer(dec, act)
    return dec

def InstanceNormLayer(layer):
    from custom_layers.normalization import StandardizationLayer
    layer = StandardizationLayer(layer, axes='spatial')
    layer = ScaleLayer(layer, shared_axes='auto')
    layer = BiasLayer(layer, shared_axes='auto')
    return layer

def g_unet_256(in_shp, is_a_grayscale, is_b_grayscale, nf=64, mul_factor=[1,2,4,8,8,8,8,8], act=tanh, dropout_p=0., bilinear_upsample=False, instance_norm=False):
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
    
    norm_layer = BatchNormLayer if not instance_norm else InstanceNormLayer
    
    # 1,2,4,8,8,8,8,8

    mf = mul_factor
    
    # in_ch x 256 x 256
    conv1 = Convolution(i, nf*mf[0])
    conv1 = norm_layer(conv1)
    x = NonlinearityLayer(conv1, nonlinearity=leaky_rectify)
    # nf x 128 x 128
    conv2 = Convolution(x, nf * mf[1])
    conv2 = norm_layer(conv2)
    x = NonlinearityLayer(conv2, nonlinearity=leaky_rectify)
    # nf*2 x 64 x 64
    conv3 = Convolution(x, nf * mf[2])
    conv3 = norm_layer(conv3)
    x = NonlinearityLayer(conv3, nonlinearity=leaky_rectify)
    # nf*4 x 32 x 32
    conv4 = Convolution(x, nf * mf[3])
    conv4 = norm_layer(conv4)
    x = NonlinearityLayer(conv4, nonlinearity=leaky_rectify)
    # nf*8 x 16 x 16
    conv5 = Convolution(x, nf * mf[4])
    conv5 = norm_layer(conv5)
    x = NonlinearityLayer(conv5, nonlinearity=leaky_rectify)
    # nf*8 x 8 x 8
    conv6 = Convolution(x, nf * mf[5])
    conv6 = norm_layer(conv6)
    x = NonlinearityLayer(conv6, nonlinearity=leaky_rectify)
    # nf*8 x 4 x 4
    conv7 = Convolution(x, nf * mf[6])
    conv7 = norm_layer(conv7)
    x = NonlinearityLayer(conv7, nonlinearity=leaky_rectify)
    # nf*8 x 2 x 2
    conv8 = Convolution(x, nf * mf[7], k=2, s=1, border_mode='valid')
    conv8 = norm_layer(conv8)
    x = NonlinearityLayer(conv8, nonlinearity=leaky_rectify)
    
    #dconv1 = Deconvolution(x, nf * 8,
    #                       k=2, s=1)
    dconv1 = Deconvolution(x, nf * mf[6], k=2, s=1)
    dconv1 = norm_layer(dconv1) #2x2
    dconv1 = DropoutLayer(dconv1, p=dropout_p)
    x = concatenate_layers([dconv1, conv7])
    x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
    # nf*(8 + 8) x 2 x 2
    dconv2 = ups(x, nf * mf[5])
    dconv2 = norm_layer(dconv2)
    dconv2 = DropoutLayer(dconv2, p=dropout_p)
    x = concatenate_layers([dconv2, conv6])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 4 x 4
    dconv3 = ups(x, nf * mf[4])
    dconv3 = norm_layer(dconv3)
    dconv3 = DropoutLayer(dconv3, p=dropout_p)
    x = concatenate_layers([dconv3, conv5])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 8 x 8
    dconv4 = ups(x, nf * mf[3])
    dconv4 = norm_layer(dconv4)
    x = concatenate_layers([dconv4, conv4])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 16 x 16
    dconv5 = ups(x, nf * mf[2])
    dconv5 = norm_layer(dconv5)
    x = concatenate_layers([dconv5, conv3])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(8 + 8) x 32 x 32
    dconv6 = ups(x, nf * mf[1])
    dconv6 = norm_layer(dconv6)
    x = concatenate_layers([dconv6, conv2])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(4 + 4) x 64 x 64
    dconv7 = ups(x, nf * mf[0])
    dconv7 = norm_layer(dconv7)
    x = concatenate_layers([dconv7, conv1])
    x = NonlinearityLayer(x, leaky_rectify)
    # nf*(2 + 2) x 128 x 128
    dconv9 = ups(x, 1 if is_b_grayscale else 3)
    # out_ch x 256 x 256
    #act = 'sigmoid' if is_binary else 'tanh'
    out = NonlinearityLayer(dconv9, act)
    
    return out


def discriminator(in_shp, is_grayscale, nf=32, act=sigmoid, mul_factor=[1,2,4,8], strides=[2,2,2,2], num_repeats=0, instance_norm=False, stride_last_conv=True):
    assert len(mul_factor) == len(strides)
    i = InputLayer((None, 1 if is_grayscale else 3, in_shp, in_shp))
    x = i
    for i in range(len(mul_factor)):
        for r in range(num_repeats+1):
            x = Convolution(x, nf*mul_factor[i], s=strides[i] if r == 0 else 1)
            x = NonlinearityLayer(x, leaky_rectify)
            if not instance_norm:
                x = BatchNormLayer(x)
            else:
                x = InstanceNormLayer(x)
    x = Convolution(x, 1, s=2 if stride_last_conv else 1)
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
    l_out = block9(256, False, False)
    print l_out.output_shape
