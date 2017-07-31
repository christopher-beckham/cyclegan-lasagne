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

from util import convert_to_rgb, compose_imgs, plot_grid

class CycleGAN():
    def _print_network(self,l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape, "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
        print "# learnable params:", count_params(layer, trainable=True)
    def __init__(self,
                 gen_fn_atob, disc_fn_a,
                 gen_params_atob, disc_params_a,
                 gen_fn_btoa, disc_fn_b,
                 gen_params_btoa, disc_params_b,
                 in_shp, is_a_grayscale, is_b_grayscale,
                 alpha_atob=100, alpha_btoa=100, opt=adam, opt_args={'learning_rate':theano.shared(floatX(1e-3))},
                 reconstruction='l1', lsgan=False, no_gan=False, verbose=True):
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale
        self.in_shp = in_shp
        self.verbose = verbose
        # get the networks for the p2p network
        gen_atob = gen_fn_atob(in_shp, is_a_grayscale, is_b_grayscale, **gen_params_atob)
        gen_btoa = gen_fn_btoa(in_shp, is_b_grayscale, is_a_grayscale, **gen_params_btoa)
        # is A real or generated?
        disc_a = disc_fn_a(in_shp, is_a_grayscale, **disc_params_a)
        # is B real or generated?
        disc_b = disc_fn_b(in_shp, is_b_grayscale, **disc_params_b)
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
        if no_gan:
            # if we turn off the GAN, we should only do cycle loss
            atob_gen_total_loss = atob_gen_cycle_loss 
        ## btoa losses
        # adversarial loss
        btoa_disc_loss = adv_loss(btoa['disc_out_real'], 1.).mean() + adv_loss(btoa['disc_out_fake'], 0.).mean()
        btoa_gen_loss = adv_loss(btoa['disc_out_fake'], 1.).mean()
        # backward cycle consistency loss
        btoa_gen_cycle_loss = T.abs_(B-btoa['cycle']).mean()
        btoa_gen_total_loss = btoa_gen_loss + alpha_btoa*btoa_gen_cycle_loss ####
        if no_gan:
            # if we turn off the GAN, we should only do cycle loss
            btoa_gen_total_loss = btoa_gen_cycle_loss
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
        if not no_gan:
            updates.update(opt(atob_disc_loss, disc_params_atob, **opt_args)) # update atob disc
        updates.update(opt(btoa_gen_total_loss, gen_params_btoa, **opt_args)) # update btoa generator
        if not no_gan:
            updates.update(opt(btoa_disc_loss, disc_params_btoa, **opt_args)) # update btoa disc
        # do da functions
        if self.verbose:
            print "creating fns..."
        fn_keys = [atob_gen_loss, atob_gen_cycle_loss, atob_disc_loss, btoa_gen_loss, btoa_gen_cycle_loss, btoa_disc_loss]
        train_fn = theano.function([A,B], fn_keys, updates=updates, on_unused_input='warn')
        loss_fn = theano.function([A,B], fn_keys, on_unused_input='warn')
        atob_fn = theano.function([A], atob['gen_out'])
        btoa_fn = theano.function([B], btoa['gen_out'])
        atob_fn_det = theano.function([A], atob['gen_out_det'])
        btoa_fn_det = theano.function([B], btoa['gen_out_det'])
        self.train_fn = train_fn
        self.loss_fn = loss_fn
        self.atob_fn = atob_fn
        self.btoa_fn = btoa_fn
        self.atob_fn_det = atob_fn_det
        self.btoa_fn_det = btoa_fn_det
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
            try:
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
                dump_train = "%s/dump_train" % out_dir
                dump_valid = "%s/dump_valid" % out_dir
                for path in [dump_train, dump_valid]:
                    if not os.path.exists(path):
                        os.makedirs(path)
                # plot nice grids
                plot_grid("%s/atob_%i.png" % (out_dir,e+1), it_val, self.atob_fn, invert=False, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
                plot_grid("%s/btoa_%i.png" % (out_dir,e+1), it_val, self.btoa_fn, invert=True, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
                # plot big pictures of predict(A) in the valid set
                #self.generate_atobs(it_train, 1, batch_size, "%s/dump_train" % out_dir, deterministic=False)
                #self.generate_atobs(it_val, 1, batch_size, "%s/dump_valid" % out_dir, deterministic=False)
                self.plot(itr=it_train, out_filename="%s/atob_%i.png" % (dump_train, e+1), out_filename_gt="%s/atob_%i_gt.png" % (dump_train, e+1), mode='atob')
                self.plot(itr=it_train, out_filename="%s/btoa_%i.png" % (dump_train, e+1), out_filename_gt="%s/btoa_%i_gt.png" % (dump_train, e+1), mode='btoa')
                self.plot(itr=it_val, out_filename="%s/atob_%i.png" % (dump_valid, e+1), out_filename_gt="%s/atob_%i_gt.png" % (dump_valid, e+1), mode='atob')
                self.plot(itr=it_val, out_filename="%s/btoa_%i.png" % (dump_valid, e+1), out_filename_gt="%s/btoa_%i_gt.png" % (dump_valid, e+1), mode='btoa')
                #filename = "%s/%s_%i.png" % (out_dir, mode, epoch)
                #filename_gt = "%s/%s_%i_gt.png" % (out_dir, mode, epoch)

                if model_dir != None and (e+1) % save_every == 0:
                    self.save_model("%s/%i.model" % (model_dir, e+1))
            except KeyboardInterrupt:
                import pdb
                pdb.set_trace()
    '''
    def generate_atobs(self, itr, num_examples, batch_size, out_dir, deterministic=True):
        if deterministic:
            atob_fn, btoa_fn = self.atob_fn_det, self.btoa_fn_det
        else:
            atob_fn, btoa_fn = self.atob_fn, self.btoa_fn
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        from skimage.io import imsave
        ctr = 0
        for n in range(num_examples // batch_size):
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
    '''
    def plot(self, itr, out_filename, out_filename_gt, grid_size=10, mode='atob', deterministic=True):
        assert mode in ['atob', 'btoa']
        if deterministic:
            atob_fn, btoa_fn = self.atob_fn_det, self.btoa_fn_det
        else:
            atob_fn, btoa_fn = self.atob_fn, self.btoa_fn
        ############
        n_channel_a = 1 if self.is_a_grayscale else 3
        n_channel_b = 1 if self.is_b_grayscale else 3
        # grid with transformed images
        in_shp = self.in_shp
        if mode == 'atob':
            # a -> b
            grid = floatX( np.zeros((in_shp*grid_size, in_shp*grid_size, n_channel_b)) )
            grid_gt = floatX( np.zeros((in_shp*grid_size, in_shp*grid_size, n_channel_a)) )
        else:
            # b -> a
            grid = floatX( np.zeros((in_shp*grid_size, in_shp*grid_size, n_channel_a)) )
            grid_gt = floatX( np.zeros((in_shp*grid_size, in_shp*grid_size, n_channel_b)) )
        this_A, this_B = itr.next()
        ctr = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if ctr == itr.bs:
                    # if we've used all the imgs in the batch, get a fresh new batch
                    this_A, this_B = itr.next()
                    ctr = 0
                if mode == 'atob':
                    target = atob_fn(this_A)
                    grid_gt[i*in_shp:(i+1)*in_shp, j*in_shp:(j+1)*in_shp, :] = convert_to_rgb(this_A[ctr], self.is_a_grayscale)
                    grid[i*in_shp:(i+1)*in_shp, j*in_shp:(j+1)*in_shp, :] = convert_to_rgb(target[ctr], self.is_b_grayscale)
                else:
                    target = btoa_fn(this_B)
                    grid_gt[i*in_shp:(i+1)*in_shp, j*in_shp:(j+1)*in_shp, :] = convert_to_rgb(this_B[ctr], self.is_b_grayscale)
                    grid[i*in_shp:(i+1)*in_shp, j*in_shp:(j+1)*in_shp, :] = convert_to_rgb(target[ctr], self.is_a_grayscale)
                ctr += 1
        from skimage.io import imsave
        if grid.shape[-1] == 1:
            grid = grid[:,:,0]
        if grid_gt.shape[-1] == 1:
            grid_gt = grid_gt[:,:,0]
        imsave(arr=grid,fname=out_filename)
        imsave(arr=grid_gt,fname=out_filename_gt)
