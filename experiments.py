from keras.preprocessing.image import ImageDataGenerator
from util import Hdf5TwoClassIterator
import shutil
import os
import sys
from cycle_gan import CycleGAN
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import floatX
import numpy as np
from architectures import p2p
import shutil
import h5py
    
if __name__ == '__main__':


    def get_dr_iterators(batch_size):
        dr_h5 = "/data/lisatmp4/beckhamc/hdf5/dr.h5"
        dataset = h5py.File(dr_h5,"r")
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        # c1 has latent factor of interest, i.e. Au
        # c2 doesn't have factor of interest, i.e. B0
        it_train = Hdf5TwoClassIterator(X=dataset['xt'], y=dataset['yt'],
                                     bs=batch_size, imgen=imgen, c1=4, c2=0,
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        it_val = Hdf5TwoClassIterator(X=dataset['xv'], y=dataset['yv'],
                                     bs=batch_size, imgen=imgen, c1=4, c2=0,
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        return it_train, it_val
            

    # i think d=0.5 is only good for a->b, for b->a it was terrible
    def dr1_l100b(mode):
        debug = True
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_dr_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256,
            disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params,
            disc_params_a=disc_params,
            
            gen_fn_btoa=p2p.g_unet_256,
            disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params,
            disc_params_b=disc_params,
            
            in_shp=256,
            is_a_grayscale=False,
            is_b_grayscale=False,
            alpha_atob=100,
            alpha_btoa=100,
            lsgan=True,
            opt=adam,
            opt_args={'learning_rate':theano.shared(floatX(1e-3))},
            no_gan=True
        )
        name = "dr1_l100b_debug"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, quick_run=debug)


    def dr1_l100c(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_dr_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256,
            disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params,
            disc_params_a=disc_params,
            
            gen_fn_btoa=p2p.g_unet_256,
            disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params,
            disc_params_b=disc_params,
            
            in_shp=256,
            is_a_grayscale=False,
            is_b_grayscale=False,
            alpha_atob=100,
            alpha_btoa=100,
            lsgan=True,
            opt=rmsprop,
            opt_args={'learning_rate':theano.shared(floatX(1e-4))},
        )
        name = "dr1_l100c"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, quick_run=debug)


    def dr1_l100b_beef1(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True, 'mul_factor':[1,2,4,8,8,8,16,16] }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_dr_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256,
            disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params,
            disc_params_a=disc_params,
            
            gen_fn_btoa=p2p.g_unet_256,
            disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params,
            disc_params_b=disc_params,
            
            in_shp=256,
            is_a_grayscale=False,
            is_b_grayscale=False,
            alpha_atob=100,
            alpha_btoa=100,
            lsgan=True,
            opt=adam,
            opt_args={'learning_rate':theano.shared(floatX(1e-3))},
        )
        name = "dr1_l100b_beef1"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, quick_run=debug)


            
    def dr1_l10b(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_dr_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256,
            disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params,
            disc_params_a=disc_params,
            
            gen_fn_btoa=p2p.g_unet_256,
            disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params,
            disc_params_b=disc_params,
            
            in_shp=256,
            is_a_grayscale=False,
            is_b_grayscale=False,
            alpha_atob=10,
            alpha_btoa=10,
            lsgan=True,
            opt=adam,
            opt_args={'learning_rate':theano.shared(floatX(1e-3))},
        )
        name = "dr1_l10b"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, quick_run=debug)


            
    def dr1_l200(mode):
        debug = True
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_dr_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256,
            disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params,
            disc_params_a=disc_params,
            
            gen_fn_btoa=p2p.g_unet_256,
            disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params,
            disc_params_b=disc_params,
            
            in_shp=256,
            is_a_grayscale=False,
            is_b_grayscale=False,
            alpha_atob=200,
            alpha_btoa=200,
            lsgan=True,
            opt=adam,
            opt_args={'learning_rate':theano.shared(floatX(2e-4))},
        )
        name = "dr1_l200"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, quick_run=debug)


            
    locals()[ sys.argv[1] ]( sys.argv[2] )
