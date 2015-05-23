#!/bin/bash

import os
import urllib2


URLS_DIR = 'urls'
TRAINING_IMAGE_DIR = 'images/raw'


def main(urls_dir, train_dir):
  # Find images we've already grabbed
  already_grabbed = set(
    handle.partition('.')[0]
    for handle in os.listdir(train_dir))

  # Grab new ones!
  skipped = 0
  grabbed = 0
  failed = 0
  for handle in os.listdir(urls_dir):
    with open(os.path.join(urls_dir, handle), 'r') as f:
      for line in f:
        iid, url = line.strip().split()
        _, _, ext = url.rpartition('.')
        if iid in already_grabbed:
          skipped += 1
        else:
          if skipped:
            print 'Skipped %d previously grabbed images.' % skipped
          skipped = 0
          try:
            print 'Grabbing %s...' % url
            outhandle = os.path.join(train_dir, '%s.%s' % (iid, ext[:3]))
            http = urllib2.urlopen(url, timeout=2)
            img = http.read()
            with open(outhandle, 'w') as out:
              out.write(img)
            grabbed += 1
            print 'Grabbed: %d; Wrote %s!' % (grabbed, outhandle)
          except Exception as e:
            failed += 1
            print 'Failed: %d; Something went wrong grabbing %s! %s' % (
              failed, url, e)


if __name__ == '__main__':
  main(URLS_DIR, TRAINING_IMAGE_DIR)
