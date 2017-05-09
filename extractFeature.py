
import os
import sys
import re
import xml.etree.ElementTree as ET
import pdb
import scipy.ndimage as ndimage
import scipy.io as sio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import glob
import time

caffe_path = '/home/harp/SaveData/Sharon/caffe/python'
sys.path.insert(0,caffe_path)
import caffe

use_gt_class = 0
size = 640,480 
maxsamples = 20  # maximum number of sampled pixels for training for each class in a single image

def load_classlabel(file):
    tree = ET.parse(file)
    classes =[]
    root = tree.getroot()
    for item in root.findall('object'):
        name = item.find('name').text
        classes.append("'"+name+"'\n")
    return classes


def print_shape(layer_name):
  f1 = net.blobs[layer_name].data[0]
  print layer_name+" shape =", f1.shape


def resize_blob(layer_name):
  # won't use this function because of memory constraint
  imgsize = size # same as input image size 
  feat = net.blobs[layer_name].data[0]
  feat_resized = ndimage.zoom(feat, (1, imgsize[1]/feat.shape[1], imgsize[0]/feat.shape[2]), order=0)
  return feat_resized


# extract sampled feature of each class from an image
def feature_out_sampled():
  imgsize = size  # same as input image size (640,480)
  feature_layers = ['conv3_3', 'conv4_3', 'conv5_3', 'fc7']
  feats = []
  scales = []
  print "Extracting FCN feature layers"
  for feat_name in feature_layers:
    print_shape(feat_name)
    feat = net.blobs[feat_name].data[0]
    feats.append(feat)
    scales.append([feat.shape[2]/float(imgsize[0]), feat.shape[1]/float(imgsize[1])])

  #TODO::
  #load ground truth label to select the target pixels
  im_path = gtDir + os.path.basename(f)[:os.path.basename(f).rfind(".")] + '.png'
  im_cls = np.asarray(Image.open(im_path))
  #im_cls = im_cls.reshape(-1,1)
  im_cls_sampled = np.empty((0), int)
 
  #print "Shuffling index of different pixels"
  classes = np.unique(im_cls)
  #print ("unique class {}".format(classes))
  ind_classes = []
  for cls in classes:
    ind = np.where(im_cls==cls)
    p = np.random.permutation(len(ind[0]))
    if cls > 0:
      num_pixel = min(maxsamples, len(ind[0]))
    else:
      num_pixel = 1
    ind_shuffled = [ind[0][p][:num_pixel], ind[1][p][:num_pixel]]
    ind_classes.append(ind_shuffled)

  #print "Generating pixelwise FCN features from ", feature_layers 
  #fig=plt.figure()
  #fig.set_tight_layout(True)
  #print ("len(ind_classes) {}".format(len(ind_classes)))
  for c in range(len(ind_classes)):
    cls = ind_classes[c] 
    #print ("len(cls) {}".format(len(cls[0])))
    for x in range(len(cls[0])):
      i = cls[0][x]
      j = cls[1][x]
      pixel_cls = im_cls[i,j][0]
      pixel_feat = np.empty((0), float)
      for ind in range(len(feature_layers)):
        feat_vector = feats[ind][:,int(i*scales[ind][1]), int(j*scales[ind][0])]
        pixel_feat=np.hstack((pixel_feat, feat_vector))
      np.savetxt(fout_feat, pixel_feat.reshape(1,-1), fmt='%.7f', delimiter=',')
      #print ("add {}".format(pixel_cls))
      im_cls_sampled = np.hstack((im_cls_sampled,pixel_cls))

    #print c, "/",len(ind_classes), "  class: ", classes[c], "   ", all_labels[classes[c]]
  
  np.savetxt(fout_cls, im_cls_sampled, fmt='%d')


# load ground truth label from SegmentationClass 

def compare_label():
  imgsize = size
  im_pred = net.blobs['score'].data[0]
  #print "im_pred.shape", im_pred.shape  # (41, 360, 640)

  #im_path = os.path.dirname(f)[:os.path.dirname(f).rfind("/")] +'/SegmentationClass/' + os.path.basename(f)[:os.path.basename(f).rfind(".")] + '.png'
  #im_cls = np.asarray(Image.open(im_path))
  #print "im_cls.shape",im_cls.shape #(360, 640)
  
  locs = [[218,221],   #object -> background
          [87,170],    #background -> object
          [227,256]]   #object -> object
  for l in locs:
    #pixel_cls = im_cls[l[0],l[1]]
    pixel_pred = im_pred[:,l[1],l[0]].reshape((1,41))
    print l
    #print "pixel ground truth label: ", pixel_cls
    print "pixel prediction scores: ", pixel_pred


  #for i in range(imgsize[0]):
  #  for j in range(imgsize[1]):
  #    pixel_cls = im_cls[j,i]
  #    if pixel_cls == 0:
  #      continue
  #    pixel_pred = im_pred[:,j,i].reshape((1,41))
  #    print "pixel ground truth label: ", pixel_cls
  #    print "pixel prediction scores: ", pixel_pred
    #return 0


# main function  
CAFFEMODEL = sys.argv[1]
caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Net('deploy.prototxt', CAFFEMODEL, caffe.TEST)

folder = sys.argv[2]
files = glob.glob(os.path.join(folder, '*.jpg'))
names = dict()
all_labels = ['background']+open('classes_unknown.txt').readlines()
num_labels = len(all_labels)

gtDir = sys.argv[3]
outDir = sys.argv[4]

if use_gt_class:
  anno_folder = sys.argv[5]

outtxt_path = outDir + 'feature'+ str(maxsamples) +'.txt'
fout_feat = open(outtxt_path, 'wb')
outtxt_path = outDir + 'label'+ str(maxsamples) +'.txt'
fout_cls = open(outtxt_path, 'wb')

for f in files:
  print ""
  print "Processing ", f
  start = time.time()

  # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe 
  im = Image.open(f)
  im.thumbnail(size)
  in_ = np.array(im, dtype=np.float32)
  in_ = in_[:,:,::-1]
  in_ -= np.array((104.00698793,116.66876762,122.67891434))
  in_ = in_.transpose((2,0,1))

  # shape for input (data blob is N x C x H x W), set data
  net.blobs['data'].reshape(1, *in_.shape)
  net.blobs['data'].data[...] = in_

  # run net and take argmax for prediction
  net.forward()

  feature_out_sampled()
  #compare_label()
  end = time.time()
  print "time: ", end-start, "s"
fout_feat.close()
fout_cls.close()

print "FCN pixel features generated at " + outtxt_path 
print "FCN pixel label saved at "+outtxt_path 


