#!/bin/python

from scipy import misc


def LoadColorAndGreyscaleImages(path):
  try:
    color = misc.imread(path)
    if len(color.shape) == 3 and color.shape[2] == 3:
      return color, misc.fromimage(misc.toimage(color), flatten=True)
    else:
      print "Skipping greyscale image."
      return None, None
  except Exception as e:
    print e
    return None, None


def DownsampledPatch(image, max_x, max_y):
  y, x = image.shape[:2]
  y_ratio = max_y / float(y)
  x_ratio = max_x / float(x)
  scale = max(x_ratio, y_ratio)
  return misc.imresize(
    misc.imresize(image, scale)[:max_y,:max_x],
    (max_x, max_y))
