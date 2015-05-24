#!/bin/python

from scipy import misc


def LoadColorAndGreyscaleImages(path):
  im = misc.imread(path)
  return im, misc.fromimage(misc.toimage(im), flatten=True)


def DownsampledPatch(image, max_x, max_y):
  y, x = image.shape[:2]
  y_ratio = max_y / float(y)
  x_ratio = max_x / float(x)
  scale = max(x_ratio, y_ratio)
  return misc.imresize(
    misc.imresize(image, scale)[:max_y,:max_x],
    (max_x, max_y))
