from lasagne.layers import *
from normalization import instance_norm

l_in = InputLayer((None,3,28,28))
l_conv = Conv2DLayer(l_in, num_filters=8, filter_size=3)

layer_ = instance_norm(l_conv)

for layer in get_all_layers(layer_):
  print layer, layer.output_shape
