#!/bin/python

import os
import sys
import time

import convnets
import train


IMDIR = "images/raw"


def TimedTrainingRun(images, net, train_fn):
  mark = time.time()
  train_fn(images)
  return time.time() - mark


def main(base_net, size, starting_batch_size, runs, reps=5):
  handles = [os.path.join(IMDIR, h) for h in os.listdir(IMDIR)]
  print "Found {l} image handles".format(l=len(handles))
  print "Loading Images..."
  images, _, _ = train.ValidationTestTrainSplit(
      handles, starting_batch_size * runs, 0, size, size)
  print "Building network..."
  theano_exprs = convnets.CreateTheanoExprs(base_net, size, size, 0.01)
  net, train_fn = theano_exprs[:2]
  convnets.PrintNetworkShape(net)
  b = starting_batch_size
  run_stats = []
  for _ in xrange(runs):
    for _ in xrange(reps):
      run_time = TimedTrainingRun(images[:b], net, train_fn)
      images_per_second = b / run_time
      run_stats.append((b, run_time, images_per_second))
      print "Processed {b} images in {r:.3f} seconds: {p:.1f} ips".format(
        b=b, r=run_time, p=images_per_second)
    b += starting_batch_size
  return run_stats


if __name__ == "__main__":
  net_name, size, starting_batch_size, runs = sys.argv[1:5]
  main(
      getattr(convnets, net_name),
      int(size),
      int(starting_batch_size),
      int(runs))
