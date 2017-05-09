#!/bin/python

import os
import sys
import numpy as np
import cv2
from PIL import Image
import glob
import matplotlib.pyplot as plt

#files = glob.glob(os.path.join("gtImages/", '*.jpg'))
files = glob.glob(os.path.join("selectedBGImages/", '*.jpg'))
num_labels = 21
for f in files:
    print ("Processing {}".format(f))
    #im = Image.open(f)
    im = cv2.imread(f)
    #im = im.convert('L')
    #classNum = int(os.path.basename(f)[:os.path.basename(f).find("-")])
    out = np.asarray(im)
    #print ("shape {}".format(out.shape))
    #print type(out)
    #print ("{}".format(out.flags))
    #out.setflags(write = 1)
    idx = np.where(out > 0)
    out[idx] = 0
    #print ("nonzero {}".format(out[np.nonzero(out)]))
    #print ("shape {} {}".format(out.shape, type(out)))

    #print ("Before classes {}".format(np.unique(np.uint8(out))))
    #print ("class is {}, file name is {}".format(classNum, f))
    #save_path = os.path.dirname(f) + "/" + os.path.basename(f)[:os.path.basename(f).rfind("_")]+'.png'
    save_path = os.path.dirname(f) + "/" + os.path.basename(f)[:os.path.basename(f).rfind(".")]+'.png'
    #plt.imsave(save_path, np.uint8(out), vmin=0, vmax = num_labels, cmap='hsv')
    cv2.imwrite(save_path,out)
    img = Image.open(save_path)
    o = np.asarray(img)
    #print ("shape {}".format(o.shape))
    print ("After classes {}".format(np.unique(np.asarray(img))))
    #break 
