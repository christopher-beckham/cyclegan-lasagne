from keras.preprocessing.image import ImageDataGenerator
from util import Hdf5TwoClassIterator, Hdf5Iterator, Hdf5InMemoryIterator
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
import util

if __name__ == '__main__':

    def get_raw_dr_iterators(batch_size, more_classes=False):
        """
        Raw images.
        """
        dr_h5 = "/data/lisatmp4/beckhamc/hdf5/dr_not_normed.h5"
        if more_classes:
            c1s = (4,3)
        else:
            c1s = (4,)
        dataset = h5py.File(dr_h5,"r")
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                   rotation_range=360, fill_mode='constant', cval=-1)
        it_train = Hdf5TwoClassIterator(X=dataset['xt'], y=dataset['yt'],
                                     bs=batch_size, imgen=imgen, c1s=c1s, c2s=(0,),
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        it_val = Hdf5TwoClassIterator(X=dataset['xv'], y=dataset['yv'],
                                     bs=batch_size, imgen=imgen, c1s=c1s, c2s=(0,),
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        return it_train, it_val
    
    def get_dr_iterators(batch_size, more_classes=False):
        """
        Normalised using Ben Graham's Gaussian technique.
        batch_size:
        more_classes:
        """
        dr_h5 = "/data/lisatmp4/beckhamc/hdf5/dr.h5"
        if more_classes:
            c1s = (4,3)
        else:
            c1s = (4,)
        dataset = h5py.File(dr_h5,"r")
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        it_train = Hdf5TwoClassIterator(X=dataset['xt'], y=dataset['yt'],
                                     bs=batch_size, imgen=imgen, c1s=c1s, c2s=(0,),
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        it_val = Hdf5TwoClassIterator(X=dataset['xv'], y=dataset['yv'],
                                     bs=batch_size, imgen=imgen, c1s=c1s, c2s=(0,),
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        return it_train, it_val


    def get_blindspot_dr_iterators(batch_size, max_imgs):
        """
        Normalised using Ben Graham's Gaussian technique.
        batch_size:
        more_classes:
        """
        dr_h5 = "/data/lisatmp4/beckhamc/hdf5/dr.h5"
        dataset = h5py.File(dr_h5,"r")
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        it_train = Hdf5OneClassDebugIterator(X=dataset['xt'], y=dataset['yt'],
                                     bs=batch_size, imgen=imgen, c1s=(0,),
                                    rnd_state=np.random.RandomState(0), max_imgs=max_imgs)
        it_val = Hdf5OneClassDebugIterator(X=dataset['xv'], y=dataset['yv'],
                                     bs=batch_size, imgen=imgen, c1s=(0,),
                                    rnd_state=np.random.RandomState(0), max_imgs=max_imgs)
        return it_train, it_val

    def get_horse_iterators(batch_size):
        dr_h5 = "/data/lisatmp4/beckhamc/hdf5/horse2zebra.h5"
        dataset = h5py.File(dr_h5,"r")
        imgen = ImageDataGenerator(horizontal_flip=True)
        it_train = Hdf5Iterator(X=dataset['trainA'], y=dataset['trainB'],
                                bs=batch_size, imgen=imgen, is_a_grayscale=False,
                                is_b_grayscale=False, is_uint8=True)
        it_val = Hdf5Iterator(X=dataset['testA'], y=dataset['testB'],
                                bs=batch_size, imgen=imgen, is_a_grayscale=False,
                                is_b_grayscale=False, is_uint8=True)
        return it_train, it_val

    def preproc(img):
        img = util.min_max_then_tanh(img)
        img = util.rnd_crop(img)
        return img
    
    def get_horse_iterators_in_memory(batch_size):
        dr_h5 = "/data/lisatmp4/beckhamc/hdf5/horse2zebra_uint8.h5"
        dataset = h5py.File(dr_h5,"r")
        imgen = ImageDataGenerator(horizontal_flip=True,
                                   preprocessing_function=preproc,
                                   data_format='channels_last')
        it_train = Hdf5InMemoryIterator(X=dataset['trainA'], y=dataset['trainB'],
                                        bs=batch_size, imgen=imgen)
        it_val = Hdf5InMemoryIterator(X=dataset['testA'], y=dataset['testB'],
                                      bs=batch_size, imgen=imgen)
        return it_train, it_val

    
    def test_iterator(mode):
        it_train, it_val = get_dr_iterators(16, norm=False)
        from skimage.io import imsave
        from util import convert_to_rgb
        aa,bb = it_train.next()
        for i in range(aa.shape[0]):
            imsave(arr=convert_to_rgb(aa[i], is_grayscale=False),
                   fname="tmp/%i.png" % i)

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
            opt_args={'learning_rate':theano.shared(floatX(2e-4))},
        )
        name = "dr1_l100b_debug2"
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
            alpha_atob=1.,
            alpha_btoa=1.,
            lsgan=True,
            opt=adam,
            opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


    def dr1_l100c_lamb10_43_rotate(mode):
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
        it_train, it_val = get_dr_iterators(bs, more_classes=True)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb10_43_rotate"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)



    def dr1_l100c_lamb10_43_9block(mode):
        ''''
        Based on training loss, may need to beef up capacity
        for this.
        '''
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {}
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 8
        it_train, it_val = get_dr_iterators(bs, more_classes=True)
        model = CycleGAN(
            gen_fn_atob=p2p.block9, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.block9, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb10_43_9block_repeat_2_bnd"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


    def dr1_l100c_lamb10_43_9block_notnorm(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {}
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 8
        it_train, it_val = get_raw_dr_iterators(bs, more_classes=True)
        model = CycleGAN(
            gen_fn_atob=p2p.block9, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.block9, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb10_43_9block_notnorm"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


            
    def dr1_l100c_lamb2_notnorm(mode):
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
        it_train, it_val = get_dr_iterators(bs, norm=False)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb2_notnorm"
        model.load_model("models/%s/10.model.bak2" % name)
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, resume=True)




    def lamb2_dd(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8,8]}
        bs = 16
        it_train, it_val = get_dr_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "lamb2_dd"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


    def lamb2_dd_g128(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':128, 'bilinear_upsample':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8,8]}
        bs = 8
        it_train, it_val = get_dr_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "lamb2_dd_g128"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


            
    def dr1_l100c_lamb2_notnorm_43(mode):
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
        it_train, it_val = get_dr_iterators(bs, norm=False, more_classes=True)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb2_notnorm_43"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, resume=True)



    def dr1_l100c_lamb2_notnorm_43_agr(mode):
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
        it_train, it_val = get_raw_dr_iterators(bs, more_classes=True)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb2_notnorm_43_agr"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, resume=True)


            


            
    def dr1_l100c_lamb2_notnorm_43_d01(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True, 'dropout':0.1 }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_dr_iterators(bs, norm=False, more_classes=True)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb2_notnorm_43_d0.1"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, resume=True)


            
    def dr1_l100c_lamb10_notnorm(mode):
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
        it_train, it_val = get_dr_iterators(bs, norm=False)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb10_notnorm"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)



            
            
    def dr1_l100c_lamb3(mode):
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
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=3., alpha_btoa=3.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "dr1_l100c_lamb3"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)

    # --------------------------------------------------------------------------------------------------

    def blindspot1_lamb2_maximg1000(mode):
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
        it_train, it_val = get_blindspot_dr_iterators(bs, max_imgs=1000)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "blindspot1_lamb2_maximg1000_repeat"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, save_every=5)

    def blindspot1_lamb10_maximg1000(mode):
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
        it_train, it_val = get_blindspot_dr_iterators(bs, max_imgs=1000)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "blindspot1_lamb10_maximg1000_repeat"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, save_every=5)


    def blindspot1_lamb2_block9_maximg1000(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 8
        it_train, it_val = get_blindspot_dr_iterators(bs, max_imgs=1000)
        model = CycleGAN(
            gen_fn_atob=p2p.block9, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.block9, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "blindspot1_lamb2_block9_maximg1000_repeat"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, save_every=5)

    def blindspot1_lamb10_block9_maximg1000(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 8
        it_train, it_val = get_blindspot_dr_iterators(bs, max_imgs=1000)
        model = CycleGAN(
            gen_fn_atob=p2p.block9, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.block9, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "blindspot1_lamb10_block9_maximg1000_repeat"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, save_every=5)

            

    # ----------------------


    def horse_lamb10(mode):
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
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb10"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


    def horse_lamb2(mode):
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
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb2"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


    def horse_lamb3(mode):
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
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=3., alpha_btoa=3.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb3"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


    def horse_lamb4(mode):
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
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=4., alpha_btoa=4.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb4"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)

    def horse_lamb4_d64(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':64, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=4., alpha_btoa=4.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb4_d64"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


    def horse_lamb10_d64(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True }
            disc_params = {'nf':64, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb10_d64"
        model.load_model("models/%s/100.model.bak" % name)
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(0,100), resume=True)

    def horse_lamb10_d64_inorm(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True, 'instance_norm':True }
            disc_params = {'nf':64, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb10_d64_inorm"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,100))
            


    def horse_lamb10_d64_inormboth(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True, 'instance_norm':True }
            disc_params = {'nf':64, 'instance_norm':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb10_d64_inormboth"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,100))



    def horse_lamb10_d64_inormboth_b1_dconv(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':False, 'instance_norm':True }
            disc_params = {'nf':64, 'instance_norm':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4)), 'beta1': 0.5},
            reconstruction='l1',
        )
        name = "horse_lamb10_d64_inormboth_b1-0.5_dconv"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,100))

    def horse_lamb10_d64_inormboth_b1_dconv_inmem(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':False, 'instance_norm':True }
            disc_params = {'nf':64, 'instance_norm':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_horse_iterators_in_memory(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4)), 'beta1': 0.5},
            reconstruction='l1',
        )
        name = "horse_lamb10_d64_inormboth_b1-0.5_dconv_inmem"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,100))

    def horse_lamb10_d64_inormboth_b1_dconv_inmem_d5(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':False, 'instance_norm':True, 'dropout_p':0.5 }
            disc_params = {'nf':64, 'instance_norm':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_horse_iterators_in_memory(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4)), 'beta1': 0.5},
            reconstruction='l1',
        )
        name = "horse_lamb10_d64_inormboth_b1-0.5_dconv_inmem_d5_joint_rc"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,200))


    def horse_lamb10_d64_inormboth_b1_dconv_inmem_d5_weakd(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':False, 'instance_norm':True, 'dropout_p':0.5 }
            disc_params = {'nf':64, 'instance_norm':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8], 'strides':[2,2,2,1], 'stride_last_conv':False}
        bs = 16
        it_train, it_val = get_horse_iterators_in_memory(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4)), 'beta1': 0.5},
            reconstruction='l1',
        )
        name = "horse_lamb10_d64_inormboth_b1-0.5_dconv_inmem_d5_joint_rc_weakd"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,200))


    def horse_block9_lamb10_d64_inormboth_b1_dconv_inmem_d5_weakd(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'instance_norm':True}
            disc_params = {'nf':64, 'instance_norm':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8], 'strides':[2,2,2,1], 'stride_last_conv':False}
        bs = 4
        it_train, it_val = get_horse_iterators_in_memory(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.block9, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.block9, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4)), 'beta1': 0.5},
            reconstruction='l1',
        )
        name = "horse_block9_lamb10_d64_inormboth_b1-0.5_dconv_inmem_d5_joint_rc_weakd"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,200))
            


    def horse_block9_64_lamb10_d64_inormboth_b1_dconv_inmem_d5_weakd(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'instance_norm':True, 'nf':64}
            disc_params = {'nf':64, 'instance_norm':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8], 'strides':[2,2,2,1], 'stride_last_conv':False}
        bs = 4
        it_train, it_val = get_horse_iterators_in_memory(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.block9, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.block9, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4)), 'beta1': 0.5},
            reconstruction='l1',
        )
        name = "horse_block9-64_lamb10_d64_inormboth_b1-0.5_dconv_inmem_d5_joint_rc_weakd"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,200))


            
            

    def horse_lamb10_d64_inormboth_b1_dconv_inmem_d5_joint_rc_bs1(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':False, 'instance_norm':True, 'dropout_p':0.5 }
            disc_params = {'nf':64, 'instance_norm':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 1
        it_train, it_val = get_horse_iterators_in_memory(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=10., alpha_btoa=10.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4)), 'beta1': 0.5},
            reconstruction='l1',
        )
        name = "horse_lamb10_d64_inormboth_b1-0.5_dconv_inmem_d5_joint_rc_bs1"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=200, out_dir="output/%s" % name, model_dir="models/%s" % name, decay_lr=(100,200))



            
    def horse_lamb2_inorm(mode):
        debug = False
        if debug:
            p2p.g_unet_256 = p2p.fake_generator
            p2p.discriminator = p2p.fake_discriminator
            gen_params = {}
            disc_params = {}
        else:
            gen_params = {'nf':64, 'bilinear_upsample':True, 'instance_norm':True }
            disc_params = {'nf':32, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]}
        bs = 16
        it_train, it_val = get_horse_iterators(bs)
        model = CycleGAN(
            gen_fn_atob=p2p.g_unet_256, disc_fn_a=p2p.discriminator,
            gen_params_atob=gen_params, disc_params_a=disc_params, 
            gen_fn_btoa=p2p.g_unet_256, disc_fn_b=p2p.discriminator,
            gen_params_btoa=gen_params, disc_params_b=disc_params,
            in_shp=256,
            is_a_grayscale=False, is_b_grayscale=False,
            alpha_atob=2., alpha_btoa=2.,
            lsgan=True,
            opt=adam, opt_args={'learning_rate':theano.shared(floatX(2e-4))},
            reconstruction='l1',
        )
        name = "horse_lamb2_inorm"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)

            
    locals()[ sys.argv[1] ]( sys.argv[2] )
