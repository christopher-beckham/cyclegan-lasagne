import h5py
import glob
from skimage.io import imread
import numpy as np

h5 = h5py.File('/data/lisatmp4/beckhamc/hdf5/horse2zebra_uint8.h5', 'w')

dirs = ['trainA', 'trainB', 'testA', 'testB']
for i in range(len(dirs)):
    files = glob.glob("horse2zebra/%s/*.jpg" % dirs[i])
    print "len of %s = %i" % (dirs[i], len(files))
    h5.create_dataset(dirs[i], shape=(len(files), 256, 256, 3), dtype='uint8')
    for j in range(len(files)):
        img = imread(files[j])
        if len(img.shape) == 2:
            # if black and white, add extra channels
            img = np.asarray([img,img,img],dtype=img.dtype).swapaxes(0,1).swapaxes(1,2)
        h5[ dirs[i] ][j] = img

h5.close()
    
