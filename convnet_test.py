#!/bin/python

import os
import sys

import numpy as np

import convnets
import train

SIZE = 80
IMDIR = "images/raw"


def main(*net_names):
  if not net_names:
    net_names = [k for k in dir(convnets) if "_NET" in k]
  handles = [os.path.join(IMDIR, h) for h in os.listdir(IMDIR)]
  np.random.shuffle(handles)
  print "Found {l} image handles".format(l=len(handles))
  for net_name in net_names:
    print "\nTesting network named {n}".format(n=net_name)
    print "\nBuilding network..."
    net_base = getattr(convnets, net_name)
    transform = net_base["transform"]
    builder = net_base["builder"]
    net, train_fn, val_fn = convnets.CreateTheanoExprs(
        builder, transform, SIZE, SIZE, 0.01)
    convnets.PrintNetworkShape(net)
    print "\nStarting Train / Validate / Test procedure..."
    batch_stats, val_stats, err, net  = train.Train(
        num_batches=4,
        validate_every_n_batches=2,
        height=SIZE,
        width=SIZE,
        batch_size=10,
        image_handles=handles,
        val_set_size=20,
        test_set_size=30,
        net=net,
        train_fn=train_fn,
        val_fn=val_fn)


if __name__ == "__main__":
  main(*sys.argv[1:])