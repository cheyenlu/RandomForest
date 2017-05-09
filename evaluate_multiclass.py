import numpy as np
from sklearn import metrics
import os
import sys
import matplotlib.pyplot as plt


def evaluate(gt_filename, label_filename, dist_filename, metric):
  labels_gt = np.genfromtxt(gt_filename, dtype=np.int, delimiter=',')
  labels    = np.genfromtxt(label_filename, dtype=np.int, delimiter=',')
  dists     = np.genfromtxt(dist_filename, dtype=np.float32, delimiter=',')

  print "labels_gt.shape", labels_gt.shape
  print "labels.shape",labels.shape
  print "dists.shape",dists.shape

  # reshape multiclass label to [num_samples, num_class]
  num_class = int(max(np.amax(labels_gt)+1, np.amax(labels)+1))
  print "num_class: ", num_class
  labels_gt_vec = np.zeros((labels_gt.shape[0], num_class),dtype=np.int)
  for i in range(labels_gt.shape[0]):
    labels_gt_vec[i, labels_gt[i]]=1
  labels_vec = np.zeros((labels.shape[0], num_class),dtype=np.int)
  for i in range(labels.shape[0]):
    labels_vec[i, labels[i]]=1

  s1=metrics.accuracy_score(labels_gt_vec, labels_vec)
  s8=metrics.confusion_matrix(labels_gt, labels)

  print "accuracy  = %f" % s1

  print "confusion matrix: "
  print s8
  plt.imshow(s8)
  plt.savefig('confusion_matrix')
  print(metrics.classification_report(labels_gt, labels))

  