#!/bin/python

import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import cPickle

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from evaluate_multiclass import evaluate


# normalize to [-1,1]
def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    v = v/norm
    v = (v-0.5)*2
    return v


# Performs Random Forest classification and save the model to a local file
if __name__ == '__main__':
  #if len(sys.argv) != 4:
  #  print "Usage: {0} feat_file label_file output_model".format(sys.argv[0])
  #  print "feat_file -- file of features"
  #  print "label_file -- file of labels"
  #  print "output_model -- path to save the svm model"
  #  exit(1)
  max_training_num = 1000
  start = time.time()

  feat_file  = sys.argv[1]
  label_file = sys.argv[2]
  output_dir = sys.argv[3]
  out_model  = output_dir + "apc40.rf.model" # sys.argv[3]
  gtfeat     = output_dir + "gt_feature.txt"
  gtlabel    = output_dir + "gt_label.txt" #sys.argv[4]
  vallabel   = output_dir + "val_label.txt" #sys.argv[6]
  valprob    = output_dir + "val_prob.txt" #sys.argv[7]

  train_svm = True  # True: train; False: test

  if train_svm:
    print "Loading feature and label data..."
    feats  = np.genfromtxt(feat_file, dtype=np.float32, delimiter=",")
    labels = np.genfromtxt(label_file, dtype=np.int) #.reshape(-1, 1)

    p = np.random.permutation(len(labels))
    #num_pixel = min(max_training_num, len(labels))
    num_pixel = len(labels)
    feats = feats[p][:num_pixel]
    labels = labels[p][:num_pixel]

    #normalize feature
    #feats = normalize(feats)

    X_train, X_test, y_train, y_test = train_test_split(
        feats, labels, test_size=0.2, random_state=0)
    print "Training num : ", X_train.shape[0], "   Label shape  : ", y_train.shape
    print "Testing  num : ", X_test.shape[0]

    # Multiclass Random Forest training and tuning params
    print "Start random forest grid search..."
    score_metric = 'accuracy'  #['precision', 'recall', 'average_precision']

    forest = RandomForestClassifier(n_estimators=25)
    param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 5, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

    # run grid search
    clf = GridSearchCV(forest, param_grid=param_grid)

    clf.fit(X_train, y_train)
    print 'Best paramenters found on training set: ', clf.best_params_
    # Best paramenters found on training set:  {'bootstrap': False, 'min_samples_leaf': 1 or 3, 'min_samples_split': 2, 'criterion': 'entropy', 'max_features': 10, 'max_depth': None}

    # Save RF model
    with open(out_model, "wb") as f:
      cPickle.dump(clf, f, cPickle.HIGHEST_PROTOCOL)
      print 'RF trained successfully! Model saved at ', out_model
    end1 = time.time()

    np.savetxt(gtfeat, X_test, fmt="%f",  delimiter=',')  # large data
    np.savetxt(gtlabel, y_test, fmt="%d", delimiter=',')
    print "Training random forest done! Time: ", end1-start, "s"

  else:
    print "Loading testing features. Loading ground truth labels for evaluation..."
    clf = cPickle.load(open(out_model,"rb"))
    X_test = np.genfromtxt(gtfeat, dtype=np.float32, delimiter=",")
    y_test = np.genfromtxt(gtlabel, dtype=np.int)

    # random forest classification
    labels_pred = clf.predict(X_test)
    labels_prob = clf.predict_proba(X_test)
    end2 = time.time()
    print "Testing random forest done! Time: ", end2-start, "s"
    #print "prediction shape: ", labels_pred.shape
    #print "distance function shape: ", labels_prob.shape

    np.savetxt(vallabel, labels_pred, fmt="%d", delimiter=',')
    np.savetxt(valprob, labels_prob, fmt="%f", delimiter=',')

    eval_metric = 'accuracy'
    evaluate(gtlabel, vallabel, valprob, eval_metric)
    print "Evaluation done!"
    end3 = time.time()
    print "Evaluation time: ", end3-end2, "s"
    print "Total time: ", end3-start, "s"


