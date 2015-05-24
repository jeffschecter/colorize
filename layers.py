#!/bin/python
#
# Adapted from the Theano convolutional net example at:
# http://deeplearning.net/tutorial/lenet.html
#
# The structure of the code is basically identical; I'm just rephrasing
# the docs to make sure I actually understand what I'm doing, getting the
# code inline with the style of the rest of the project, doing a bit of
# refactoring to allow more specification of hyperparameters, and adatping
# to the use case of image colorization.

import numpy as np
import theano

from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

def Bound(nonlin, total_fan):
  """Returns a reasonable bound to initialize a layer's weights.

  Args:
    nonlin: (T.elemwise.Elemwise) Nonlinearity applied to the layer's output.
    total_fan: (int) Sum of the layer's fan in and fan out.

  Returns:
    (float) The bound.
  """
  if nonlin == T.nnet.sigmoid:
    return 4 * np.sqrt(6.0 / total_fan)
  elif nonlin == T.tanh:
    return np.sqrt(6.0 / total_fan)
  else:
    print "Nonlinearity not recognized; bound defaulting to sqrt(6 / fan)."
    return np.sqrt(6.0 / total_fan)


# --------------------------------------------------------------------------- #
# Layer classes                                                               #
# --------------------------------------------------------------------------- #

class MaxPoolLayer(object):
  """Max pooling layer."""

  def __init__(self, rng, inp, filter_shape, image_shape, poolsize, nonlin):
    """Create a max pooling layer.

    Args:
      rng: (np.random.RandomState) Used to initialize weights.
      inp: (T.dtensor4) Input images of shape image_shape.
      filter_shape: (tuple of len 4) (num filters, num input feature maps,
        filter height, filter width).
      image_shape: (tuple of len 4) (batch size, input feature maps, image
        height, image width).
      poolsize: (tuple of len 2) (num rows, num cols) The downsampling factor.
      nonlin: (T.elemwise.Elemwise) The nonlinearity applied to layer output.
    """
    print "Creating a max pooling layer..."
    assert image_shape[1] == filter_shape[1]
    self.input = inp

    # Initialize W with random weights.
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] *
               np.prod(filter_shape[2:]) /
               np.prod(poolsize))
    W_bound = Bound(nonlin, fan_in + fan_out)
    self.W = theano.shared(
      np.asarray(
        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
        dtype=theano.config.floatX),
      borrow=True) # TODO look this up.

    # Each output feature map has its own bias.
    b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
    self.b = theano.shared(value=b_values, borrow=True)

    # Apply filters to input.
    conv_out = conv.conv2d(
      input=inp,
      filters=self.W,
      filter_shape=filter_shape,
      image_shape=image_shape) 

    # Downsample the output of each filter separately.
    pooled_out = downsample.max_pool_2d(
      input=conv_out,
      ds=poolsize,
      ignore_border=True)

    self.params = [self.W, self.b]
    self.output = nonlin(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class HiddenLayer(object):
  """Fully connected hidden layer."""

  def __init__(self, rng, inp, n_in, n_out, nonlin):
    """Create a fully connected hidden layer.

    Args:
      rng: (np.random.RandomState) Used to initialize weights.
      inp: (T.dtensor4) Input images of shape image_shape.
      n_in: (int) Fan in.
      n_out: (int) Fan out.
      nonlin: (T.elemwise.Elemwise) The nonlinearity applied to layer output.
    """
    print "Creating a hidden layer..."
    self.input = inp

    # Initialize weight matrix.
    W_bound = Bound(nonlin, n_in + n_out)
    W_values = np.asarray(
      rng.uniform(
          low=-W_bound,
          high=W_bound,
          size=(n_in, n_out)),
      dtype=theano.config.floatX)
    self.W = theano.shared(value=W_values, name='W', borrow=True)

    # Initialize biases.
    b_values = np.zeros((n_out,), dtype=theano.config.floatX)
    self.b = theano.shared(value=b_values, name='b', borrow=True)

    self.params = [self.W, self.b]
    self.output = nonlin(T.dot(inp, self.W) + self.b)


class ProjectionLayer(object):
  """Layer that changes the dimensionality of input images."""

  def __init__(self, rng, inp, input_shape, output_shape, nonlin):
    """Create a layer that projects an image into a different dimensionality.

    Each pixel in the output map is connected only to corresponding pixels
    (across all channels) in the input map.

    If the input image is smaller than the output image, each input pixil will
    instead be connected to all pixels in a region of output. For instance, for
    a 25px-by-25px input projecting to a 100px-by-100px output, each input pixel
    will connect to 16 output pixels in a 4px-by-4px region. The reverse will be
    the case if the input image is larger than the output image.

    Args:
      rng: (np.random.RandomState) Used to initialize weights.
      inp: (T.dtensor4) Input images of shape input_shape.
      input_shape: (tuple of len 4) (batch size, input image channels,
        image height, image width)
      output_shape: (tuple of len 4) (batch size, output image channels,
        image height, image width)
      nonlin: (T.elemwise.Elemwise) The nonlinearity applied to layer output.
    """
    print "Creating a projection layer..."
    self.input = inp

    # Initialize W with random weights.
    fan_in = (input_shape[1] *
              max(1, output_shape[2] / input_shape[2]) *
              (1, output_shape[3] / input_shape[3]))
    fan_out = output_shape[1]
    bound = Bound(nonlin, fan_in + fan_out)
    W_values = np.zeros()
    self.W
    self.b

    self.params = [self.W, self.b]
    self.output = nonlin(T.dot(inp, self.W) + self.b)
