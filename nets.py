#!/bin/python

import numpy as np
import theano

from theano import tensor as T

import image
import layers


class ColorizerNet(object):
  """Adds color to gresycale images."""

  def __init__(self, rng, image_shape, batch_size, nkerns, filter_span,
               poolsize, nonlin):
    """Builds a model to colorize greyscale images.

         [GREYSCALE INPUT IMAGE]
           |                |
    [CONV+MP LAYER]         |
           |                |
    [CONV+MP LAYER]         |
           |                |
     [PROJ LAYER]      [PROJ LAYER]
           |                |
          [COLOR OUTPUT IMAGE]

    Args:
      rng: (np.random.RandomState) Used to initialize weights.
      image_shape: (tuple of len 2) (x pixels, y pixels)
      batch_size: (int) Training images per batch.
      nkerns: (tuple of ints of len 2) Number of kernels in the first and
        second max pooling layers, respectively.
      filter_span: (int) Height/width of kernel filters.
      poolsize: (int) Downsampling factor.
      nonlin: (T.elemwise.Elemwise) The nonlinearity applied to the output of
        each layer. T.tanh is standard.
    """
    print 'Building the model...'

    # Examples are gresycale images; targets are full color images
    x = T.matrix('x')
    y = T.tensor3('y')

    # Inpput to first layer is greyscale, ie a single feature map.
    x_shape, y_shape = image_shape
    layer0_input = x.reshape((batch_size, 1, x_shape, y_shape))

    # First convolutional layer.
    x_shape0 = (x_shape - filter_span + 1) / poolsize
    y_shape0 = (y_shape - filter_span + 1) / poolsize
    layer0 = layers.MaxPoolLayer(
      rng=rng,
      inp=layer0_input,
      filter_shape=(nkerns[0], 1, filter_span, filter_span),
      image_shape=(batch_size, 1, x_shape, y_shape),
      poolsize=(poolsize, poolsize),
      nonlin=nonlin)

    # Second convolutional layer.
    x_shape1 = (x_shape0 - filter_span + 1) / poolsize
    y_shape1 = (y_shape0 - filter_span + 1) / poolsize
    layer1 = layers.MaxPoolLayer(
      rng=rng,
      inp=layer0.output,
      filter_shape=(nkerns[1], nkerns[0], filter_span, filter_span),
      image_shape=(batch_size, nkerns[0], x_shape0, y_shape0),
      poolsize=(poolsize, poolsize),
      nonlin=nonlin)

    # Third layer is a fully connected hidden layer.
    layer2_input = layer1.output.flatten(2)
    layer2 = layers.HiddenLayer(
        rng=rng,
        inp=layer2_input,
        n_in=nkerns[1] * x_shape1 * y_shape1,
        n_out=np.prod(image_shape) * 3,
        nonlin=nonlin)

    #TODO  everything below here is wrong
    # Fourth layer 
    return

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )