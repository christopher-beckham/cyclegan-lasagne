import os
import random
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def _get_slices(length, bs):
    slices = []
    b = 0
    while True:
        if b*bs >= length:
            break
        slices.append( slice(b*bs, (b+1)*bs) )
        b += 1
    return slices

'''
def iterate_hdf5(imgen=None, is_a_grayscale=True, is_b_grayscale=False, is_uint8=True):
    def _iterate_hdf5(X_arr, y_arr, bs, rnd_state=np.random.RandomState(0)):
        assert X_arr.shape[0] == y_arr.shape[0]
        while True:
            # we manage two slices, since this is meant to be used for
            # cycle gan, which doesn't need corresponding [a,b] pairs
            slices_x = _get_slices(X_arr.shape[0], bs)
            slices_y = _get_slices(y_arr.shape[0], bs)
            if rnd_state != None:
                rnd_state.shuffle(slices_x)
                rnd_state.shuffle(slices_y)
            for slice_x, slice_y in zip(slices_x, slices_y):
                this_X, this_Y = X_arr[slice_x].astype("float32"), y_arr[slice_y].astype("float32")
                # TODO: only compatible with theano
                this_X = this_X.swapaxes(3,2).swapaxes(2,1)
                this_Y = this_Y.swapaxes(3,2).swapaxes(2,1)
                # normalise A and B if these are in the range [0,255]
                if is_uint8:
                    this_X = (this_X / 255.0) if is_a_grayscale else (this_X - 127.5) / 127.5
                    this_Y = (this_Y / 255.0) if is_b_grayscale else (this_Y - 127.5) / 127.5
                # if we passed an image generator, augment the images
                if imgen != None:
                    seed = rnd_state.randint(0, 100000)
                    this_X = imgen.flow(this_X, None, batch_size=bs, seed=seed).next()
                    this_Y = imgen.flow(this_Y, None, batch_size=bs, seed=seed).next()              
                yield this_X, this_Y
    return _iterate_hdf5

# this just wraps the above functional iterator
class Hdf5Iterator():
    def __init__(self, X, y, bs, imgen, is_a_grayscale, is_b_grayscale, is_uint8=True):
        """
        :X: in our case, the heightmaps
        :y: in our case, the textures
        :bs: batch size
        :imgen: optional image data generator
        :is_a_binary: if the A image is binary, we have to divide
         by 255, otherwise we scale to [-1, 1] using tanh scaling
        :is_b_binary: same as is_a_binary
        """
        assert X.shape[0] == y.shape[0]
        self.fn = iterate_hdf5(imgen, is_a_grayscale, is_b_grayscale, is_uint8)(X, y, bs)
        self.N = X.shape[0]
    def __iter__(self):
        return self
    def next(self):
        return self.fn.next()
'''


class Hdf5TwoClassIterator():
    def __init__(self, X, y, bs, imgen, c1s, c2s, tanh_norm=True, rnd_state=np.random.RandomState(0), debug=False):
        """
        :X: in our case, the heightmaps
        :y: in our case, the textures
        :bs:
        :imgen:
        :c1s:
        :c2s
        """
        # build the list of indices corresponding to c1, and c2
        self.idxs_c1 = []
        self.idxs_c2 = []
        for c1 in c1s:
            self.idxs_c1 += np.where(y[:]==c1)[0].tolist()
        for c2 in c2s:
            self.idxs_c2 += np.where(y[:]==c2)[0].tolist()
        if debug:
            print "idxs_c1", idxs_c1, "length =", len(idxs_c1)
            print "idxs_c2", idxs_c2, "length =", len(idxs_c2)
        # save slices
        self.slices_for_c1 = _get_slices(len(self.idxs_c1), bs)
        self.slices_for_c2 = _get_slices(len(self.idxs_c2), bs)
        # book-keeping
        self.N = min(len(self.idxs_c1), len(self.idxs_c2))
        self.bs = bs
        self.X = X
        self.rnd_state = rnd_state
        self.tanh_norm = tanh_norm
        self.imgen = imgen
        self.fn = self._iterate()
    def _iterate(self):
        while True:
            if self.rnd_state != None:
                self.rnd_state.shuffle(self.slices_for_c1)
                self.rnd_state.shuffle(self.slices_for_c2)
            for elem1,elem2 in zip(self.slices_for_c1, self.slices_for_c2):
                this_X, this_Y = self.X[ self.idxs_c1[elem1] ], self.X[ self.idxs_c2[elem2] ]
                if this_X.shape[0] != this_Y.shape[0]:
                    # batch size mis-match, go to start of while loop
                    break
                if self.tanh_norm:
                    # go between [0,1], then go to [-1, 1]
                    norm_params = {'axis': (1,2,3), 'keepdims':True}
                    this_X = (this_X - np.min(this_X,**norm_params)) / ( np.max(this_X,**norm_params) - np.min(this_X,**norm_params) )
                    this_X = (this_X - 0.5) / 0.5
                    this_Y = (this_Y - np.min(this_Y,**norm_params)) / ( np.max(this_Y,**norm_params) - np.min(this_Y,**norm_params) )
                    this_Y = (this_Y - 0.5) / 0.5
                    # if we passed an image generator, augment the images
                if self.imgen != None:
                    seed = self.rnd_state.randint(0, 100000)
                    this_X = self.imgen.flow(this_X, None, batch_size=self.bs, seed=seed).next()
                    this_Y = self.imgen.flow(this_Y, None, batch_size=self.bs, seed=seed).next()
                yield this_X, this_Y
    def __iter__(self):
        return self
    def next(self):
        return self.fn.next()


        
"""
Code borrowed from Pedro Costa's vess2ret repo:
https://github.com/costapt/vess2ret
"""

def convert_to_rgb(img, is_grayscale=False):
    """Given an image, make sure it has 3 channels and that it is between 0 and 1."""
    if len(img.shape) != 3:
        raise Exception("""Image must have 3 dimensions (channels x height x width). """
                        """Given {0}""".format(len(img.shape)))
    img_ch, _, _ = img.shape
    if img_ch != 3 and img_ch != 1:
        raise Exception("""Unsupported number of channels. """
                        """Must be 1 or 3, given {0}.""".format(img_ch))
    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)
    if not is_grayscale:
        imgp = imgp * 127.5 + 127.5
        imgp /= 255.
    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)


def compose_imgs(a, b, is_a_grayscale=True, is_b_grayscale=False):
    """Place a and b side by side to be plotted."""
    ap = convert_to_rgb(a, is_grayscale=is_a_grayscale)
    bp = convert_to_rgb(b, is_grayscale=is_b_grayscale)
    if ap.shape != bp.shape:
        raise Exception("""A and B must have the same size. """
                        """{0} != {1}""".format(ap.shape, bp.shape))
    # ap.shape and bp.shape must have the same size here
    h, w, ch = ap.shape
    composed = np.zeros((h, 2*w, ch))
    composed[:, :w, :] = ap
    composed[:, w:, :] = bp
    return composed

def plot_grid(out_filename, itr, out_fn, invert, is_a_grayscale, is_b_grayscale, N=4):
    plt.figure(figsize=(10, 6))
    if invert:
        # swap the image modes around
        is_a_grayscale, is_b_grayscale = is_b_grayscale, is_a_grayscale
    for i in range(N*N):
        a, b = itr.next()
        if invert:
            b, a = a, b
        if out_fn != None:
            bp = out_fn(a)
        else:
            bp = b
        img = compose_imgs(a[0], bp[0], is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
        plt.subplot(N, N, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(out_filename)
    plt.clf()
    # Make sure all the figures are closed.
    plt.close('all')


    
if __name__ == '__main__':

    import h5py

    def get_dr_iterators(batch_size):
        dr_h5 = "/data/lisatmp4/beckhamc/hdf5/dr.h5"
        dataset = h5py.File(dr_h5,"r")
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        # c1 has latent factor of interest, i.e. Au
        # c2 doesn't have factor of interest, i.e. B0
        it_train = Hdf5TwoClassIterator(X=dataset['xt'], y=dataset['yt'],
                                     bs=batch_size, imgen=imgen, c1s=(3,4), c2s=(0,),
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        it_val = Hdf5TwoClassIterator(X=dataset['xv'], y=dataset['yv'],
                                     bs=batch_size, imgen=imgen, c1s=(3,4), c2s=(0,),
                                     rnd_state=np.random.RandomState(0),
                                     tanh_norm=True)
        return it_train, it_val

    it_train, it_val = get_dr_iterators(8)
    print it_train.next()
    print it_val.next()
