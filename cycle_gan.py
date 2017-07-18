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
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import nolearn
from keras_ports import ReduceLROnPlateau
import pickle
import sys
import gzip

from util import iterate_hdf5, Hdf5Iterator, convert_to_rgb, compose_imgs, plot_grid

class CycleGAN():
    def _print_network(self,l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape, "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
        print "# learnable params:", count_params(layer, trainable=True)
    def __init__(self,
                 gen_fn_p2p, disc_fn_p2p,
                 gen_params_p2p, disc_params_p2p,
                 in_shp, is_a_grayscale, is_b_grayscale,
                 alpha_atob=100, alpha_btoa=100, opt=adam, opt_args={'learning_rate':theano.shared(floatX(1e-3))},
                 reconstruction='l1', lsgan=False, verbose=True):
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale
        self.in_shp = in_shp
        self.verbose = verbose
        # get the networks for the p2p network
        gen_atob = gen_fn_p2p(in_shp, is_a_grayscale, is_b_grayscale, **gen_params_p2p)
        gen_btoa = gen_fn_p2p(in_shp, is_b_grayscale, is_a_grayscale, **gen_params_p2p)
        # is A real or generated?
        disc_a = disc_fn_p2p(in_shp, is_a_grayscale, **disc_params_p2p)
        # is B real or generated?
        disc_b = disc_fn_p2p(in_shp, is_b_grayscale, **disc_params_p2p)        
        if verbose:
            self._print_network(gen_atob)
            self._print_network(disc_a)
        A = T.tensor4('A') # A
        B = T.tensor4('B') # B
        # atob stuff
        atob = {'gen': gen_atob, 'disc': disc_b}
        atob['gen_out'] = get_output(atob['gen'], A) # generate B from A
        atob['disc_out_real'] = get_output(atob['disc'], B) # feeding real B into disc
        atob['disc_out_fake'] = get_output(atob['disc'], atob['gen_out']) # feeding fake B into disc
        atob['gen_out_det'] = get_output(atob['gen'], A, deterministic=True) # generate B from A
        # btoa stuff
        btoa = {'gen': gen_btoa, 'disc': disc_a}
        btoa['gen_out'] = get_output(btoa['gen'], B) # generate A from B
        btoa['disc_out_real'] = get_output(btoa['disc'], A) # feeding real A into disc
        btoa['disc_out_fake'] = get_output(btoa['disc'], btoa['gen_out']) # feeding fake A into disc
        btoa['gen_out_det'] = get_output(btoa['gen'], B, deterministic=True) # generate A from B
        # cycle stuff
        atob['cycle'] = get_output(btoa['gen'], atob['gen_out']) # A -> B, then B -> A
        btoa['cycle'] = get_output(atob['gen'], btoa['gen_out']) # B -> A, then A -> B
        # save dicts
        self.atob = atob
        self.btoa = btoa
        # loss functions
        if lsgan:
            adv_loss = squared_error
        else:
            adv_loss = binary_crossentropy
        if self.verbose:
            print "creating losses..."
        # we assume that the discriminator D(x)/D(y) is the prob that x/y is real
        ## atob losses
        # adversarial loss
        atob_disc_loss = adv_loss(atob['disc_out_real'], 1.).mean() + adv_loss(atob['disc_out_fake'], 0.).mean()
        atob_gen_loss = adv_loss(atob['disc_out_fake'], 1.).mean()
        # forward cycle consistency loss
        atob_gen_cycle_loss = T.abs_(A-atob['cycle']).mean()
        atob_gen_total_loss = atob_gen_loss + alpha_atob*atob_gen_cycle_loss
        ## btoa losses
        # adversarial loss
        btoa_disc_loss = adv_loss(btoa['disc_out_real'], 1.).mean() + adv_loss(btoa['disc_out_fake'], 0.).mean()
        btoa_gen_loss = adv_loss(btoa['disc_out_fake'], 1.).mean()
        # backward cycle consistency loss
        btoa_gen_cycle_loss = T.abs_(B-btoa['cycle']).mean()
        btoa_gen_total_loss = btoa_gen_loss + alpha_btoa*btoa_gen_cycle_loss ####
        ## params
        # atob params
        gen_params_atob = get_all_params(atob['gen'], trainable=True)
        disc_params_atob = get_all_params(atob['disc'], trainable=True)
        # pix2pix params
        gen_params_btoa = get_all_params(btoa['gen'], trainable=True)
        disc_params_btoa = get_all_params(btoa['disc'], trainable=True)
        # do da updates
        if self.verbose:
            print "creating updates..."
        updates = opt(atob_gen_total_loss, gen_params_atob, **opt_args) # update atob generator
        updates.update(opt(atob_disc_loss, disc_params_atob, **opt_args)) # update atob disc
        updates.update(opt(btoa_gen_total_loss, gen_params_btoa, **opt_args)) # update btoa generator
        updates.update(opt(btoa_disc_loss, disc_params_btoa, **opt_args)) # update btoa disc
        # do da functions
        if self.verbose:
            print "creating fns..."
        train_fn = theano.function([A,B], [atob_gen_loss, atob_gen_cycle_loss, atob_disc_loss,
                                             btoa_gen_loss, btoa_gen_cycle_loss, btoa_disc_loss], updates=updates, on_unused_input='warn')
        loss_fn = theano.function([A,B], [atob_gen_loss, atob_gen_cycle_loss, atob_disc_loss,
                                             btoa_gen_loss, btoa_gen_cycle_loss, btoa_disc_loss], on_unused_input='warn')
        atob_fn = theano.function([A], atob['gen_out'])
        btoa_fn = theano.function([B], btoa['gen_out'])
        atob_fn_det = theano.function([A], atob['gen_out_det'])
        btoa_fn_det = theano.function([B], btoa['gen_out_det'])
        self.train_fn = train_fn
        self.loss_fn = loss_fn
        self.atob_fn = atob_fn
        self.btoa_fn = btoa_fn
        self.lr = opt_args['learning_rate']
        self.train_keys = ['atob_gen', 'atob_cycle', 'atob_disc', 'btoa_gen', 'btoa_cycle', 'btoa_disc']
    def save_model(self, filename):
        with gzip.open(filename, "wb") as g:
            pickle.dump({
                'atob': {'gen': get_all_param_values(self.atob['gen']), 'disc': get_all_param_values(self.atob['disc'])},
                'btoa': {'gen': get_all_param_values(self.btoa['gen']), 'disc': get_all_param_values(self.btoa['disc'])}
            }, g, pickle.HIGHEST_PROTOCOL )
    def load_model(self, filename):
        """
        filename:
        mode: what weights should we load? E.g. `both` = load
          weights for both p2p and dcgan.
        """
        with gzip.open(filename) as g:
            dd = pickle.load(g)
            set_all_param_values(self.atob['gen'], dd['atob']['gen'])
            set_all_param_values(self.atob['disc'], dd['atob']['disc'])                
            set_all_param_values(self.btoa['gen'], dd['btoa']['gen'])
            set_all_param_values(self.btoa['disc'], dd['btoa']['disc'])
    def train(self, it_train, it_val, batch_size, num_epochs, out_dir, model_dir=None, save_every=10, resume=False, reduce_on_plateau=False, schedule={}, quick_run=False):
        def _loop(fn, itr):
            rec = [ [] for i in range(len(self.train_keys)) ]
            for b in range(itr.N // batch_size):
                A_batch, B_batch = it_train.next()
                results = fn(A_batch,B_batch)
                for i in range(len(results)):
                    rec[i].append(results[i])
                if quick_run:
                    break
            return tuple( [ np.mean(elem) for elem in rec ] )
        header = ["epoch"]
        for key in self.train_keys:
            header.append("train_%s" % key)
        for key in self.train_keys:
            header.append("valid_%s" % key)
        header.append("lr")
        header.append("time")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if model_dir != None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.verbose:
            try:
                from nolearn.lasagne.visualize import draw_to_file
                draw_to_file(get_all_layers(self.atob['gen']), "%s/gen_atob.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.atob['disc']), "%s/disc_atob.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.btoa['gen']), "%s/gen_btoa.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.btoa['disc']), "%s/disc_btoa.png" % out_dir, verbose=True)
            except:
                pass
        f = open("%s/results.txt" % out_dir, "a" if resume else "wb")
        if not resume:
            f.write(",".join(header)+"\n"); f.flush()
            print ",".join(header)
        cb = ReduceLROnPlateau(self.lr,verbose=self.verbose)
        if self.verbose:
            print "training..."
        for e in range(num_epochs):
            if e+1 in schedule:
                self.lr.set_value( schedule[e+1] )
            out_str = []
            out_str.append(str(e+1))
            t0 = time()
            # training
            results = _loop(self.train_fn, it_train)
            for i in range(len(results)):
                out_str.append(str(results[i]))
            if reduce_on_plateau:
                cb.on_epoch_end(np.mean(recon_losses), e+1)
            # validation
            results = _loop(self.loss_fn, it_val)
            for i in range(len(results)):
                out_str.append(str(results[i]))
            out_str.append(str(cb.learning_rate.get_value()))
            out_str.append(str(time()-t0))
            out_str = ",".join(out_str)
            print out_str
            f.write("%s\n" % out_str); f.flush()
            # plot nice grids
            plot_grid("%s/atob_%i.png" % (out_dir,e+1), it_val, self.atob_fn, invert=False, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
            plot_grid("%s/btoa_%i.png" % (out_dir,e+1), it_val, self.btoa_fn, invert=True, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
            # plot big pictures of predict(A) in the valid set
            self.generate_atobs(it_train, 1, "%s/dump_train" % out_dir, deterministic=False)
            self.generate_atobs(it_val, 1, "%s/dump_valid" % out_dir, deterministic=False)
            if model_dir != None and (e+1) % save_every == 0:
                self.save_model("%s/%i.model" % (model_dir, e+1))
    def generate_atobs(self, itr, num_examples, out_dir, deterministic=True):
        if deterministic:
            atob_fn, btoa_fn = self.atob_fn_det, self.btoa_fn_det
        else:
            atob_fn, btoa_fn = self.atob_fn, self.btoa_fn
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        from skimage.io import imsave
        ctr = 0
        for n in range(num_batches):
            this_a, this_b = itr.next()
            pred_b = atob_fn(this_a) # A --> B
            pred_a = btoa_fn(this_b) # B --> A
            for i in range(pred_a.shape[0]):
                pred_b_processed = convert_to_rgb(pred_b[i], is_grayscale=self.is_b_grayscale)
                pred_a_processed = convert_to_rgb(pred_a[i], is_grayscale=self.is_a_grayscale)
                imsave(fname="%s/%i.a.png" % (out_dir, ctr), arr=pred_a_processed)
                imsave(fname="%s/%i.b.png" % (out_dir, ctr), arr=pred_b_processed)
                ctr += 1
                if ctr == num_examples:
                    break

def get_iterators(dataset, batch_size, is_a_grayscale, is_b_grayscale, da=True):
    dataset = h5py.File(dataset,"r")
    if da:
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=360, fill_mode="reflect")
    else:
        imgen = ImageDataGenerator()
    it_train = Hdf5Iterator(dataset['xt'], dataset['yt'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
    it_val = Hdf5Iterator(dataset['xv'], dataset['yv'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
    return it_train, it_val
                
if __name__ == '__main__':
    from architectures import p2p
    import shutil

    def copy_if_not_exist(dest_file, from_file):
        dirname = os.path.dirname(dest_file)
        if not os.path.isfile(dest_file):
            print "copying data from %s to %s" % (from_file, dest_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            shutil.copy(src=from_file, dst=dest_file)

    # desert h5 file
    # no valid set for this one
    #dest_file, from_file = "/Tmp/beckhamc/hdf5/textures_v2_brown500.h5", "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
    dest_file, from_file = "/Tmp/beckhamc/hdf5/textures_v2_brown500_with_valid.h5", "/data/lisa/data/cbeckham/textures_v2_brown500_with_valid.h5"    
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
            
    def test1(mode):
        nf_p, nf_d = 64, 64 #64...
        model = CycleGAN(
            gen_fn_p2p=p2p.g_unet,
            disc_fn_p2p=p2p.discriminator,
            gen_params_p2p={'nf':nf_p, 'act':tanh, 'num_repeats':0, 'bilinear_upsample':True},
            disc_params_p2p={'nf':nf_d, 'bn':False, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]},
            in_shp=512,
            is_a_grayscale=True,
            is_b_grayscale=False,
            lsgan=True,
            opt=rmsprop,
            opt_args={'learning_rate':theano.shared(floatX(1e-4))},
        )
        bs = 4
        it_train, it_val = get_iterators(dest_file, bs, True, False, True)
        name = "deleteme"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)

    # alpha=1 does well but the outputs can look 'chromatic'
    # maybe we should find a good value in [1,100]
    # (100 weakens the generator too much)
    #
    # NOTE: the training is pix2pix style, we get pairs [a,b]...????????
    def test2_alpha50(mode):
        model = CycleGAN(
            gen_fn_p2p=p2p.g_unet,
            disc_fn_p2p=p2p.discriminator,
            gen_params_p2p={'nf':64, 'num_repeats':0, 'bilinear_upsample':True},
            disc_params_p2p={'nf':32, 'bn':False, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]},
            in_shp=512,
            is_a_grayscale=True,
            is_b_grayscale=False,
            alpha_atob=60,
            alpha_btoa=10,
            lsgan=True,
            opt=rmsprop,
            opt_args={'learning_rate':theano.shared(floatX(1e-4))},
        )
        # alpha=10 is good for b->a
        # alpha=30 is still kinda weird.... for a->b
        bs = 4
        it_train, it_val = get_iterators(dest_file, bs, True, False, True)
        name = "deleteme2_withvalid_atob60_btoa10"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)




    def test2_atob70_btoa10(mode):
        model = CycleGAN(
            gen_fn_p2p=p2p.g_unet,
            disc_fn_p2p=p2p.discriminator,
            gen_params_p2p={'nf':64, 'num_repeats':0, 'bilinear_upsample':True},
            disc_params_p2p={'nf':32, 'bn':False, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]},
            in_shp=512,
            is_a_grayscale=True,
            is_b_grayscale=False,
            alpha_atob=70,
            alpha_btoa=10,
            lsgan=True,
            opt=rmsprop,
            opt_args={'learning_rate':theano.shared(floatX(1e-4))},
        )
        # alpha=10 is good for b->a
        # alpha=30 is still kinda weird.... for a->b
        bs = 4
        it_train, it_val = get_iterators(dest_file, bs, True, False, True)
        name = "deleteme2_withvalid_atob70_btoa10"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)

    def test2_atob75_btoa10(mode):
        model = CycleGAN(
            gen_fn_p2p=p2p.g_unet,
            disc_fn_p2p=p2p.discriminator,
            gen_params_p2p={'nf':64, 'num_repeats':0, 'bilinear_upsample':True},
            disc_params_p2p={'nf':32, 'bn':False, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]},
            in_shp=512,
            is_a_grayscale=True,
            is_b_grayscale=False,
            alpha_atob=75,
            alpha_btoa=10,
            lsgan=True,
            opt=rmsprop,
            opt_args={'learning_rate':theano.shared(floatX(1e-4))},
        )
        # alpha=10 is good for b->a
        # alpha=30 is still kinda weird.... for a->b
        bs = 4
        it_train, it_val = get_iterators(dest_file, bs, True, False, True)
        name = "deleteme2_withvalid_atob75_btoa10"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)


    def test2_atob10_btoa10_repp(mode):
        model = CycleGAN(
            gen_fn_p2p=p2p.g_unet,
            disc_fn_p2p=p2p.discriminator,
            gen_params_p2p={'nf':64, 'num_repeats':0, 'bilinear_upsample':True},
            disc_params_p2p={'nf':32, 'bn':False, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]},
            in_shp=512,
            is_a_grayscale=True,
            is_b_grayscale=False,
            alpha_atob=10,
            alpha_btoa=10,
            lsgan=True,
            opt=adam,
            opt_args={'learning_rate':theano.shared(floatX(2e-5))},
        )
        bs = 1
        it_train, it_val = get_iterators(dest_file, bs, True, False, True)
        name = "deleteme2_withvalid_atob10_btoa10_repp"
        model.load_model("models/%s/160.model.bak" % name)
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name,
                        schedule={100: 2e-6}, resume=True)

            
    locals()[ sys.argv[1] ]( sys.argv[2] )
