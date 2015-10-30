#!/bin/python

import os

import numpy as np

import convnet

SIZE = 50
IMDIR = "images/raw"


def main():
  handles = [os.path.join(IMDIR, h) for h in os.listdir(IMDIR)]
  np.random.shuffle(handles)
  print "Found {l} image handles".format(l=len(handles))
  net, train_fn, val_fn = convnet.CreateTheanoExprs(SIZE, SIZE, 0.01)
  convnet.PrintShape(net)
  batch_stats, val_stats, err, net  = convnet.Train(
      num_batches=6,
      validate_every_n_batches=2,
      height=SIZE,
      width=SIZE,
      batch_size=100,
      image_handles=handles,
      val_set_size=100,
      test_set_size=100,
      net=net,
      train_fn=train_fn,
      val_fn=val_fn)


if __name__ == "__main__":
  main()