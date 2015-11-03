#!/bin/python

import ctypes
import multiprocessing
import time

import numpy as np
from scipy import misc

import theano
from theano import tensor as T

import image


# --------------------------------------------------------------------------- #
# Image loading.                                                              #
#                                                                             #
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

def SharedArray(shape, ctype):
  shared_array_base = multiprocessing.Array(ctype, np.product(shape))
  return np.ctypeslib.as_array(shared_array_base.get_obj()).reshape(shape)


def LoadImages(handles, height, width, batch_size, shared_images,
               random=True, timer=None):
  processed = 0
  loaded = 0
  imsize = height * width * 3
  num_handles = len(handles)
  mark = time.time()
  while loaded < batch_size:
    ix = np.random.randint(0, num_handles) if random else processed
    processed += 1
    try:
      im = misc.imread(handles[ix])
      if not (len(im.shape) == 3 and im.shape[2] == 3):
        continue
      im = image.DownsampledPatch(im, width, height)
    except Exception as e:
      continue
    shared_images[loaded] = 256 - im
    loaded += 1
  load_time = time.time() - mark
  if timer:
    timer.value = load_time
  return processed


def ValidationTestTrainSplit(handles, val_set_size, test_set_size,
                             height, width):
  np.random.shuffle(handles)

  # Validation
  val_images = np.zeros((val_set_size, height, width, 3))
  offset = LoadImages(
      handles, height, width, val_set_size, val_images,
      random=False)

  # Testing
  test_images = np.zeros((test_set_size, height, width, 3))
  offset += LoadImages(
      handles[offset:], height, width, test_set_size, test_images,
      random=False)

  # Training
  train_handles = handles[offset:]

  return val_images, test_images, train_handles


# --------------------------------------------------------------------------- #
# Training & testing utilities.                                               #
# --------------------------------------------------------------------------- #

def Minibatches(images, batch_size):
  for i in range(0, len(images) - batch_size + 1, batch_size):
    yield images[i:i + batch_size]


def Test(batch_size, test_images, net, val_fn):
  print "\nTesting..."
  test_errs = []
  iterator = Minibatches(test_images, batch_size)
  mark = time.time()
  for images in iterator:
    _, err = val_fn(images)
    test_errs.append(err)
  test_time = time.time() - mark
  test_err = np.mean(test_errs)
  print ("Tested on {s} images in {t:.2f} seconds. Error = {e:.4f}.").format(
      s=len(test_images), t=test_time, e=test_err)
  return test_err


def Train(num_batches, validate_every_n_batches,
          height, width, batch_size, reps_per_batch,
          image_handles, val_set_size, test_set_size,
          net, train_fn, val_fn):
  # This works with any network and training regimen where the input and
  # targets are both functions of the same color source image.

  # Check args
  if val_set_size % batch_size != 0 or test_set_size % batch_size != 0:
    raise ValueError(
      "Validation and Test set sizes must be whole-number multiples "
      "of batch size.")

  # Split image handles in train, test, and validation sets
  print "Loading validation and testing images..."
  val_images, test_images, train_handles = ValidationTestTrainSplit(
    image_handles, val_set_size, test_set_size, height, width)

  # Memory space to shared with image loader running in background process
  image_load_timer = multiprocessing.Value('d', 0.0)
  shared_memory = SharedArray((batch_size, height, width, 3), ctypes.c_uint8)
  LoadImages(
      train_handles, height, width, batch_size, shared_memory,
      timer=image_load_timer)

  # Record keeping
  batch_stats = []
  validation_stats = []
  batch_err = 1
  batch_time = 0

  # Iterate through training batches.
  # When using the GPU, loading images from disk to RAM is a hell of a lot
  # slower than training the net on an image. To compensate, we repeat each
  # batch to be several copies of itself
  print "Starting training..."
  for b in xrange(num_batches):
    print ("Training batch {b} of {r} x {s} images. "
           "Last time = {t:.2f} seconds. "
           "Last load time = {l:.2f} seconds. "
           "Last error = {e:.5f}.").format(
              b=b, r=reps_per_batch, s=batch_size, t=batch_time,
              l=image_load_timer.value, e=batch_err)
    images = np.array(shared_memory)
    stacked = np.repeat(images, reps_per_batch, 0)
    mark = time.time()
    image_loader_process = multiprocessing.Process(
        target=LoadImages,
        args=[train_handles, height, width, batch_size, shared_memory],
        kwargs={"timer": image_load_timer})
    image_loader_process.start()
    batch_err = train_fn(images).mean()
    batch_time = time.time() - mark

    # Validation
    # TODO merge with testing logic, since it's basically identical
    if (b + 1) % validate_every_n_batches == 0:
      print "\nValidating..."
      val_errs = []
      iterator = Minibatches(val_images, batch_size)
      mark = time.time()
      for images in iterator:
        _, val_err = val_fn(images)
        val_errs.append(val_err)
      val_time = time.time() - mark
      validation_stats.append((b, batch_err, np.mean(val_errs), val_time))
      print ("Validated on {s} images in {t:.2f} seconds. "
             "Error = {e:.5f}.\n").format(
                s=len(val_images), t=val_time, e=np.mean(val_errs))

    # So we know the next batch of training images is ready
    image_loader_process.join()
    batch_stats.append((b, batch_err, batch_time, image_load_timer.value))

  test_err = Test(batch_size, test_images, net, val_fn)
  return batch_stats, validation_stats, test_err, net


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
  
