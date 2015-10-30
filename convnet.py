#!/bin/python

import multiprocessing
import os
import random
import numpy as np
import random
import scipy as sp
import time

import lasagne
import theano
import theano.tensor as T

from scipy import misc


# --------------------------------------------------------------------------- #
# Define the network.                                                         #
# --------------------------------------------------------------------------- #
  
def ScaledSigmoid(beta):
  def Closure(x):
    return beta * T.nnet.sigmoid(x)
  return Closure


def BuildNet(input_var, height, width):
  # Inputs are greyscale images.
  l_in = lasagne.layers.InputLayer(
      shape=(None, height, width),
      input_var=input_var)

  # Shuffle them into 1-channel images. 
  l_inshuf = lasagne.layers.DimshuffleLayer(
      l_in,
      (0, 'x', 1, 2))
  
  # Apply several convolutional layers, padding at each step to
  # maintain original image size. We first use a large number
  # of kernels and ReLUs for feature discovery.
  l_conv1 = lasagne.layers.Conv2DLayer(
      l_inshuf,
      num_filters=12,
      filter_size=(5, 5),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
  l_conv2 = lasagne.layers.Conv2DLayer(
      l_conv1,
      num_filters=5,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
  
  # Last convolutional layer collapses back to 3 kernels, which
  # should represent luminosity scaling factors for R, G, and B
  # channels. We use a scaled sigmoid that produces outputs between
  # 0 and 3, which is what we observe as a typical range of
  # luminosity scaling factors.
  l_conv3 = lasagne.layers.Conv2DLayer(
      l_conv2,
      num_filters=3,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=ScaledSigmoid(3),
      W=lasagne.init.GlorotUniform())
  
  # Flip the index of the channel so that outputs are in the proper
  # format for scipy color images: (height, width, rgb)
  l_outshuf = lasagne.layers.DimshuffleLayer(
      l_conv3,
      (0, 2, 3, 1))
  #l_out = ProportionNormalizationLayer(l_conv3)
  return l_outshuf


def CreateTheanoExprs(height, width, learning_rate):
  # Our target_var contains raw target images, but we're not actually
  # training on raw activation values. What we're trying to discover are
  # scaling factors that need to be applied to greyscale luminosities for
  # each channel to reconstruct the original image.
  target_var = T.tensor4("targets")
  target_ratios = target_var / (target_var.mean(axis=3, keepdims=True) + 1)

  # Inputs are greyscale images, which we can compute from the target full
  # color images.
  input_var = (target_var[:, :, 0] * 0.299 +
               target_var[:, :, 1] * 0.587 +
               target_var[:, :, 2] * 0.114)
  
  # Build network.
  net = BuildNet(input_var, height, width)

  # Loss expression.
  # Since we don't have stochastic dropout, we can use the same loss
  # expr for training and validation. If we want to add a dropout layer,
  # then we need a separate loss expression for validation where stochastic
  # elements are explicitly frozen & dropout is disabled.
  prediction = lasagne.layers.get_output(net)
  loss = lasagne.objectives.squared_error(prediction, target_ratios).mean()

  # Weight updates during training.
  params = lasagne.layers.get_all_params(net, trainable=True)
  updates = lasagne.updates.nesterov_momentum(
      loss, params, learning_rate=learning_rate, momentum=0.9)

  # Theano function to train a mini-batch.
  train_fn = theano.function(
      [target_var],
      loss,
      updates=updates,
      name="Train")

  # Theano function to evaluate / validate on an input.
  # The difference between this and the training function is that the
  # test / validation function does not apply weight updates.
  val_fn = theano.function(
      [target_var],
      [prediction, loss],
      name="Evaluate")
  
  return net, train_fn, val_fn


# --------------------------------------------------------------------------- #
# Load images from disk into memoery in a background thread on CPU while the  #
# GPU works on training the network.                                          #
#                                                                             #
# The strategy:                                                               #
#   1. Create a shared memory array to hold image data                        #
#   2. Initialize it with image data for the first runthrough.                #
#   3. Spawn background process to load training images into shared memory    #
#   4. In the foreground, update net on training data                         #
#   5. Join the background loader process                                     #
#   6. GOTO 3                                                                 #
# --------------------------------------------------------------------------- #

def LoadImages(handles, height, width, batch_size, shared_images):
  loaded = 0
  imsize = height * width * 3
  while loaded < batch_size:
    try:
      im = misc.imread(np.random.choice(handles))
      if not (len(im.shape) == 3 and im.shape[2] == 3):
        continue
    except Exception as e:
      if type(e) == KeyboardInterrupt:
        raise e
      continue
    offseet = loaded * imsize
    shared_images[offset:offset + imsize] = im.flatten()
    loaded += 1


# --------------------------------------------------------------------------- #
# Training & testing utilities.                                               #
# --------------------------------------------------------------------------- #

def Minibatches(images, batch_size):
  for i in range(0, len(images) - batch_size + 1, batch_size):
    yield images[i:i + batch_size]


def Test(batch_size, test_images, net, val_fn):
  print "\nTesting!..."
  test_errs = []
  iterator = Minibatches(test_images, batch_size)
  mark = time.time()
  for images in iterator:
    _, err = val_fn(images)
    test_errs.append(err)
  test_time = time.time() - mark
  test_err = np.mean(test_errs)
  print ("Testing completed in {t:.2f} seconds. "
         "Test error = {e:.4f}.").format(t=test_time, e=test_err)
  return test_err


def Train(num_batches, validate_every_n_batches, height, width, batch_size,
          train_handles, val_images, test_images,
          net, train_fn, val_fn):
  # Images will be loaded asynchronously by another core into shared memory
  flat_shared_memory = multiprocessing.Array(
      "H", height * width * 3 * batch_size)
  # Load up an initial batch of training images
  LoadImages(train_handles, height, width, batch_size, flat_shared_memory)

  # Record keeping
  batch_stats = []
  validation_stats = []
  batch_err = 1
  batch_time = 0

  for b in xrange(num_batches):
    print ("Training batch {b} of {s} images. "
           "Last time = {t} seconds. "
           "Last error = {e:.5f}.".format(
              b=b, s=batch_size, t=batch_time, e=batch_err)
    loader_process = multiprocessing.Process(
        target=LoadImages,
        args=[train_handles, height, width, batch_size, flat_shared_memory])
    images = np.array(flat_shared_memory).reshape(
        (batch_size, height, width, 3))
    mark = time.time()
    batch_err = train_fn(images)
    batch_time = time.time() - mark
    batch_stats.append((b, batch_err, batch_time))

    # Validation
    if (b + 1) % validate_every_n_batches == 0:
      print "\nValidating..."
      val_errs = []
      iterator = Minibatches(val_images, batch_size)
      mark = time.time()
      for images in iterator:
        _, val_err = val_fn(images)
        val_errs.append(val_err)
      val_time = time.time() - mark
      validation_stats.append(b, batch_err, np.mean(val_errs), val_time)
      print "Validated on {s} images in {t} seconds. Error = {e:.5f}.\n".format(
          s=len(val_images), t=val_time, e=np.mean(val_errs))

    # Sowe know the next batch of training images is ready
    loader_process.join()

  test_err = Test(batch_size, test_images, net, val_fn)
  return batch_stats, validation_stats, test_err, net
  