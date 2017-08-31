# -*- coding: utf-8 -*-

"""
The :class:`LocalResponseNormalization2DLayer
<lasagne.layers.LocalResponseNormalization2DLayer>` implementation contains
code from `pylearn2 <http://github.com/lisa-lab/pylearn2>`_, which is covered
by the following license:


Copyright (c) 2011--2014, Université de Montréal
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import theano
import theano.tensor as T

from lasagne import init
from lasagne import nonlinearities
from lasagne.layers.special import ScaleLayer, BiasLayer

from lasagne.layers import Layer


__all__ = [
    "StandardizationLayer",
    "instance_norm",
    "layer_norm",
]


class StandardizationLayer(Layer):
    """
    This layer implements the normalization of the input layer's
    outputs across the specified axes:

    .. math::
        y_i = \\frac{x_i - \\mu_i}{\\sqrt{\\sigma_i^2 + \\epsilon}}

    That is, each input feature (or input pixel) :math:`\\x_i`
    is normalized to zero mean and unit variance.
    The mean :math:`\\mu_i` and variance
    :math:`\\sigma_i^2` is calculated across the specified axes.
    In contrast to batch normalization, the mean and
    variance is not restricted to be defined across examples,
    so the same operation can be applied during training and testing.
    The advantages of using this implementation over e.g.
    :class:`BatchNormLayer` with adapted axes arguments, are its
    independence of the input size, as no parameters are learned and stored.
    :class:`StandardizationLayer` can be employed to realize
    different normalization schemes such as instance normalization [1]
    and layer normalization [2], for both of which convenience functions
    (:func:`instance_norm` and :func:`layer_norm`) are available.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the first two:
        this will normalize over all spatial dimensions for
        convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
        
    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience functions :func:`instance_norm` :func:`layer_norm`
    modify an existing layer to insert instance normalization or
    layer normalization in front of its nonlinearity.

    See also
    --------
    instance_norm : Convenience function to apply instance normalization to a layer
    References
    layer_norm : Convenience function to apply layer normalization to a layer
    References

    ----------
    .. [1] Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016):
           Instance Normalization: The Missing Ingredient for Fast Stylization.
           https://arxiv.org/pdf/1607.08022.pdf.

    .. [2] Ba, J., Kiros, J., & Hinton, G. (2016):
           Layer normalization. arXiv preprint arXiv:1607.06450.
    """
    def __init__(self, incoming, axes='spatial', epsilon=1e-4, **kwargs):
        super(StandardizationLayer, self).__init__(incoming, **kwargs)

        if axes == 'spatial':
            # default: normalize over spatial dimensions only,
            # separate for each instance in the batch
            axes = tuple(range(2, len(self.input_shape)))

        elif axes == 'features':
            # normalize over features and spatial dimensions,
            # separate for each instance in the batch
            axes = tuple(range(1, len(self.input_shape)))

        elif isinstance(axes, int):
            axes = (axes,)

        self.axes = axes
        self.epsilon = epsilon

    def get_output_for(self, input, **kwargs):

        mean = input.mean(self.axes, keepdims=True)
        std = T.sqrt(input.var(self.axes, keepdims=True) + self.epsilon)

        return (input - mean) / std


def instance_norm(layer, learn_scale=True, learn_bias=True, **kwargs):
    """
	Apply instance normalization to an existing layer. This is a convenience
	function modifying an existing layer to include instance normalization: It
	will steal the layer's nonlinearity if there is one (effectively
	introducing the normalization right before the nonlinearity), remove
	the layer's bias if there is one (because it would be redundant), and add
	a :class:`StandardizationLayer` and :class:`NonlinearityLayer` on top.
	Depending on the given arguments, an additional :class:`ScaleLayer` and
	:class:`BiasLayer` will be inserted inbetween.

	Parameters
	----------
	layer : A :class:`Layer` instance
		The layer to apply the normalization to; note that it will be
		irreversibly modified as specified above
	**kwargs
		Any additional keyword arguments are passed on to the
		:class:`StandardizationLayer` constructor.

	Returns
	-------
	StandardizationLayer or ScaleLayer or BiasLayer or NonlinearityLayer instance
		An instance normalization layer stacked on the given modified `layer`,
		or a scale layer or a bias layer or a nonlinearity layer stacked on top of
		the respectively present additional layers depending on whether
		`layer` was nonlinear and on the arguments given to :func:`instance_norm`.

	Examples
	--------
	Just wrap any layer into a :func:`instance_norm` call on creating it:

	>>> from lasagne.layers import InputLayer, Conv2DLayer, instance_norm
	>>> from lasagne.nonlinearities import rectify
	>>> l1 = InputLayer((10, 3, 28, 28))
	>>> l2 = instance_norm(Conv2DLayer(l1, num_filters=64, filter_size=3,
	nonlinearity=rectify))

	This introduces instance normalization right before its nonlinearity:

	>>> from lasagne.layers import get_all_layers
	>>> [l.__class__.__name__ for l in get_all_layers(l2)]
	['InputLayer', 'Conv2DLayer', 'StandardizationLayer', 'ScaleLayer',
    'BiasLayer', 'NonlinearityLayer']
	"""
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity

    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None

    in_name = (kwargs.pop('name', None) or
               (getattr(layer, 'name', None) and layer.name + '_in'))

    layer = StandardizationLayer(layer, axes='spatial', name=in_name, **kwargs)
    if learn_scale:
        layer = ScaleLayer(layer, shared_axes='auto')
    if learn_bias:
        layer = BiasLayer(layer, shared_axes='auto')

    if nonlinearity is not None:
        from lasagne.layers import NonlinearityLayer
        nonlin_name = in_name and in_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer


def layer_norm(layer, **kwargs):
    """
	Apply layer normalization to an existing layer. This is a convenience
	function modifying an existing layer to include layer normalization: It
	will steal the layer's nonlinearity if there is one (effectively
	introducing the normalization right before the nonlinearity), remove
	the layer's bias if there is one, and add
	a :class:`StandardizationLayer`,  :class:`ScaleLayer`,  :class:`BiasLayer`,
	and :class:`NonlinearityLayer` on top.

	Parameters
	----------
	layer : A :class:`Layer` instance
		The layer to apply the normalization to; note that it will be
		irreversibly modified as specified above
	**kwargs
		Any additional keyword arguments are passed on to the
		:class:`InstanceNormLayer` constructor.

	Returns
	-------
	StandardizationLayer or NonlinearityLayer instance
		A layer normalization layer stacked on the given modified `layer`,
		or a nonlinearity layer stacked on top of both
		if `layer` was nonlinear.

	Examples
	--------
	Just wrap any layer into a :func:`instance_norm` call on creating it:

	>>> from lasagne.layers import InputLayer, DenseLayer, layer_norm
	>>> from lasagne.nonlinearities import rectify
	>>> l1 = InputLayer((10, 28))
	>>> l2 = layer_norm(DenseLayer(l1, num_units=64, nonlinearity=rectify))

	This introduces layer normalization right before its nonlinearity:

	>>> from lasagne.layers import get_all_layers
	>>> [l.__class__.__name__ for l in get_all_layers(l2)]
	['InputLayer', 'DenseLayer', 'StandardizationLayer', 'ScaleLayer',
    'BiasLayer', 'NonlinearityLayer']
	"""
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity

    in_name = (kwargs.pop('name', None) or
               (getattr(layer, 'name', None) and layer.name + '_in'))

    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None

    layer = StandardizationLayer(layer, axes='features', name=in_name, **kwargs)
    layer = ScaleLayer(layer, shared_axes='auto')
    layer = BiasLayer(layer, shared_axes='auto')

    if nonlinearity is not None:
        from lasagne.layers import NonlinearityLayer
        nonlin_name = in_name and in_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer
