#!/bin/python

import numpy as np

import lasagne
import theano
from theano import tensor as T


PX_LUM_MEAN = 138.17
PX_LUM_STD = 66.55


# --------------------------------------------------------------------------- #
# Utils.                                                                      #
# --------------------------------------------------------------------------- #

class BaseNet(object):

  def __init__(self, builder, transform, train_args={}):
    self.builder = builder
    self.transform = transform
    self.train_args = train_args


def ScaledSigmoid(beta):
  def Sig(x):
    return beta * T.nnet.sigmoid(x)
  return Sig

def ScaledTanh(beta):
  def Tanh(x):
    return beta * T.tanh(x)
  return Tanh


def Grayscale(image):
  return ((image[:, :, :, 0] * 0.299 +
           image[:, :, :, 1] * 0.587 +
           image[:, :, :, 2] * 0.114) - PX_LUM_MEAN) / PX_LUM_STD


def CreateTheanoExprs(base_net, height, width, learning_rate):
  # Our target_var contains raw target images.
  target_var = T.tensor4("targets", dtype=theano.config.floatX)
  transformed_target = base_net.transform(target_var)

  # Inputs are greyscale images, which we can compute from the target full
  # color images.
  input_var = Grayscale(target_var)
  
  # Build network.
  net = base_net.builder(input_var, height, width)

  # Loss expressions.
  prediction = lasagne.layers.get_output(net)
  loss = lasagne.objectives.squared_error(
      prediction, transformed_target).mean()

  test_prediction = lasagne.layers.get_output(net, deterministic=True)
  test_loss = lasagne.objectives.squared_error(
      test_prediction, transformed_target).mean()

  # Weight updates during training.
  params = lasagne.layers.get_all_params(net, trainable=True)
  updates = lasagne.updates.nesterov_momentum(
      loss, params, learning_rate=learning_rate, momentum=0.9)

  # Theano function to train a mini-batch.
  train_fn = theano.function(
      [target_var],
      loss,
      updates=updates,
      name="Train",
      allow_input_downcast=True)

  # Theano function to evaluate / validate on an input.
  # The difference between this and the training function is that the
  # test / validation function does not apply weight updates, and any dropout
  # or other stochastic elements are frozen.
  val_fn = theano.function(
      [target_var],
      [test_prediction, test_loss],
      name="Evaluate",
      allow_input_downcast=True)
  
  return net, train_fn, val_fn, prediction, target_var, transformed_target


def PrintNetworkShape(net):
    print "{layer} {inshape} => {outshape}".format(
        layer=net.__class__.__name__,
        inshape=getattr(net, "input_shape", ""),
        outshape=getattr(net, "output_shape", ""))
    input_layer = getattr(net, "input_layer", None)
    if input_layer:
        PrintNetworkShape(input_layer)


# --------------------------------------------------------------------------- #
# Model inspection.                                                           #
# --------------------------------------------------------------------------- #

def Evaluator(prediction_var, target_var, transformed_target):
  predictor = theano.function(
    [target_var],
    [prediction_var, transformed_target])

  def Evaluate(image_or_images):
    shp = image_or_images.shape
    if len(shp) == 3:
      h, w, ch = shp
      images = image_or_images.reshape((1, h, w, ch))
    else:
      images = image_or_images
    return predictor(images)

  return Evaluate

  
def SaveNet(net, path):
  np.savez(path, *lasagne.layers.get_all_param_values(net))


def LoadSavedNet(base_net, height, width, npz_path, learning_rate=0.01):
  theano_exprs = CreateTheanoExprs(
      base_net, height, width, learning_rate)
  net = theano_exprs[0]
  with np.load(npz_path) as f:
    param_values = [f['arr_%d' % i] for i in xrange(len(f.files))]
  lasagne.layers.set_all_param_values(net, param_values)
  return theano_exprs

# --------------------------------------------------------------------------- #
# Takes a greyscale image. Learns to guess for each pixel the ratio of the    #
# brightness of each color channel to the luminosity of the greyscale pixel.  #
# --------------------------------------------------------------------------- #

def BuildLuminosityRatioNet(input_var, height, width):
  # Inputs are greyscale images.
  l_in = lasagne.layers.InputLayer(
      shape=(None, height, width),
      input_var=input_var)

  # Shuffle them into 1-channel images. 
  l_inshuf = lasagne.layers.DimshuffleLayer(
      l_in,
      (0, "x", 1, 2))
  
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


LUMINOSITY_RATIO_NET = BaseNet(
    BuildLuminosityRatioNet,
    lambda t: t / (t.mean(axis=3, keepdims=True) + 1))


# --------------------------------------------------------------------------- #
# Takes a greyscale image. Learns to guess for each pixel the value of each   #
# color channel in the original color source image.                           #
# --------------------------------------------------------------------------- #

def BuildColorizerNet(input_var, height, width):
  # Inputs are greyscale images.
  l_in = lasagne.layers.InputLayer(
      shape=(None, height, width),
      input_var=input_var)

  # Shuffle them into 1-channel images. 
  l_inshuf = lasagne.layers.DimshuffleLayer(
      l_in,
      (0, "x", 1, 2))
  
  # Apply several convolutional layers. Start with a large number of kernels
  # with a broad receptive field. Gradually narrow down the number of chanels.
  l_conv1 = lasagne.layers.Conv2DLayer(
      l_inshuf,
      num_filters=24,
      filter_size=(7, 7),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
  l_conv2 = lasagne.layers.Conv2DLayer(
      l_conv1,
      num_filters=24,
      filter_size=(5, 5),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
  l_conv3 = lasagne.layers.Conv2DLayer(
      l_conv2,
      num_filters=12,
      filter_size=(5, 5),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
  l_conv4 = lasagne.layers.Conv2DLayer(
      l_conv3,
      num_filters=6,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())

  # Dropout
  l_dropout = lasagne.layers.DropoutLayer(
      l_conv4,
      p=0.5)

  # Output a 3-channel image with RGB values between 0 and 1.
  l_conv5 = lasagne.layers.Conv2DLayer(
      l_dropout,
      num_filters=3,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=lasagne.nonlinearities.sigmoid)

  # Flip the index of the channel so that outputs are in the proper
  # format for scipy color images: (height, width, rgb)
  l_outshuf = lasagne.layers.DimshuffleLayer(
      l_conv5,
      (0, 2, 3, 1))
  return l_outshuf


COLORIZER_NET = BaseNet(BuildColorizerNet, lambda t: 1 - (t / 255))


# --------------------------------------------------------------------------- #
# As above, but narrower / less wasteful receptive fields.                    #
# --------------------------------------------------------------------------- #

def BuildNarrowColorizerNet(input_var, height, width):
  # Inputs are greyscale images.
  l_in = lasagne.layers.InputLayer(
      shape=(None, height, width),
      input_var=input_var)

  # Shuffle them into 1-channel images. 
  l_inshuf = lasagne.layers.DimshuffleLayer(
      l_in,
      (0, "x", 1, 2))
  
  # Apply several convolutional layers. Start with a large number of kernels
  # with a broad receptive field. Gradually narrow down the number of chanels.
  l_conv1 = lasagne.layers.Conv2DLayer(
      l_inshuf,
      num_filters=24,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
  l_conv2 = lasagne.layers.Conv2DLayer(
      l_conv1,
      num_filters=24,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
  l_conv3 = lasagne.layers.Conv2DLayer(
      l_conv2,
      num_filters=12,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())
  l_conv4 = lasagne.layers.Conv2DLayer(
      l_conv3,
      num_filters=6,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform())

  # Dropout
  l_dropout = lasagne.layers.DropoutLayer(
      l_conv4,
      p=0.5)

  # Output a 3-channel image with RGB values between 0 and 1.
  l_conv5 = lasagne.layers.Conv2DLayer(
      l_dropout,
      num_filters=3,
      filter_size=(3, 3),
      pad="same",
      nonlinearity=lasagne.nonlinearities.sigmoid)

  # Flip the index of the channel so that outputs are in the proper
  # format for scipy color images: (height, width, rgb)
  l_outshuf = lasagne.layers.DimshuffleLayer(
      l_conv5,
      (0, 2, 3, 1))
  return l_outshuf


NARROW_COLORIZER_NET = BaseNet(
    BuildNarrowColorizerNet,
    lambda t: 1 - (t / 255))


# --------------------------------------------------------------------------- #
# Takes a greyscale image. Learns to reconstruct the mean and standard        #
# deviation of the color histogram for each color channel in the original     #
# source image.                                                               #
# --------------------------------------------------------------------------- #

# All in RGB order
IM_MEAN_COLOR_MEANS = np.array([133.07, 139.37, 153.86])
IM_MEAN_COLOR_STDS = np.array([39.012, 37.22, 43.59])
IM_STD_COLOR_MEANS = np.array([55.46, 54.28, 55.18])
IM_STD_COLOR_STDS = np.array([16.85, 16.22, 18.63])


def ColorStatsForImages(ims):
  # Input ims is a 4d stack of 3-channel images.
  # Output should be a 2d stack with one vector of length 6 per input image.
  chan = ims.reshape((ims.shape[0], ims.shape[1] * ims.shape[2], ims.shape[3]))
  ch_mean = (chan.mean(axis=1) - IM_MEAN_COLOR_MEANS) / IM_MEAN_COLOR_STDS
  ch_std = (chan.std(axis=1) - IM_STD_COLOR_MEANS) / IM_STD_COLOR_STDS
  return T.concatenate([ch_mean, ch_std], axis=1)


def BuildColorStatsNet(input_var, height, width):
  # Inputs are greyscale images.
  l_in = lasagne.layers.InputLayer(
      shape=(None, height, width),
      input_var=input_var)

  # Shuffle them into 1-channel images.
  l_inshuf = lasagne.layers.DimshuffleLayer(
      l_in,
      (0, "x", 1, 2))

  # Apply several convolutional layers and max pooling layers.
  l_conv1 = lasagne.layers.Conv2DLayer(
      l_inshuf,
      num_filters=12,
      filter_size=(5, 5),
      nonlinearity=lasagne.nonlinearities.leaky_rectify,
      W=lasagne.init.GlorotUniform())
  l_pool1 = lasagne.layers.MaxPool2DLayer(
      l_conv1,
      pool_size=(2, 2))
  l_conv2 = lasagne.layers.Conv2DLayer(
      l_pool1,
      num_filters=5,
      filter_size=(3, 3),
      nonlinearity=lasagne.nonlinearities.leaky_rectify,
      W=lasagne.init.GlorotUniform())
  l_pool2 = lasagne.layers.MaxPool2DLayer(
      l_conv2,
      pool_size=(2, 2))

  # Fully connected hidden layer.
  l_hidden = lasagne.layers.DenseLayer(
      l_pool2,
      num_units=100,
      nonlinearity=lasagne.nonlinearities.leaky_rectify,
      W=lasagne.init.GlorotUniform())

  # Output is a vector of length 6 for each image in the batch.
  l_out = lasagne.layers.DenseLayer(
      l_hidden,
      num_units=6,
      nonlinearity=ScaledTanh(3))
  return l_out


COLOR_STATS_NET = BaseNet(BuildColorStatsNet, ColorStatsForImages)


# --------------------------------------------------------------------------- #
# As above, but deeeeperrrr.                                                  #
# --------------------------------------------------------------------------- #

def BuildDeepColorStatsNet(input_var, height, width):
  # Inputs are greyscale images.
  l_in = lasagne.layers.InputLayer(
      shape=(None, height, width),
      input_var=input_var)

  # Shuffle them into 1-channel images.
  l_inshuf = lasagne.layers.DimshuffleLayer(
      l_in,
      (0, "x", 1, 2))

  # Two convolutionl layers followed by a pooling layer.
  l_conv1 = lasagne.layers.Conv2DLayer(
      l_inshuf,
      num_filters=24,
      filter_size=(5, 5),
      nonlinearity=lasagne.nonlinearities.leaky_rectify,
      W=lasagne.init.GlorotUniform())
  l_conv2 = lasagne.layers.Conv2DLayer(
      l_conv1,
      num_filters=12,
      filter_size=(5, 5),
      nonlinearity=lasagne.nonlinearities.leaky_rectify,
      W=lasagne.init.GlorotUniform())
  l_pool1 = lasagne.layers.MaxPool2DLayer(
      l_conv2,
      pool_size=(2, 2))

  # Two more conv layers followed by a pooling layer.
  l_conv3 = lasagne.layers.Conv2DLayer(
      l_pool1,
      num_filters=6,
      filter_size=(3, 3),
      nonlinearity=lasagne.nonlinearities.leaky_rectify,
      W=lasagne.init.GlorotUniform())
  l_conv4 = lasagne.layers.Conv2DLayer(
      l_conv3,
      num_filters=6,
      filter_size=(3, 3),
      nonlinearity=lasagne.nonlinearities.leaky_rectify,
      W=lasagne.init.GlorotUniform())
  l_pool2 = lasagne.layers.MaxPool2DLayer(
      l_conv4,
      pool_size=(2, 2))

  # Dropout
  l_dropout = lasagne.layers.DropoutLayer(
      l_pool2,
      p=0.5)

  # Fully connected hidden layer.
  l_hidden = lasagne.layers.DenseLayer(
      l_dropout,
      num_units=100,
      nonlinearity=lasagne.nonlinearities.leaky_rectify,
      W=lasagne.init.GlorotUniform())

  # Output is a vector of length 6 for each image in the batch.
  l_out = lasagne.layers.DenseLayer(
      l_hidden,
      num_units=6,
      nonlinearity=ScaledTanh(3))
  return l_out


DEEP_COLOR_STATS_NET = BaseNet(BuildDeepColorStatsNet, ColorStatsForImages)