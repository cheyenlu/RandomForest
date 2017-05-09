#from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.color import gray2rgb
from skimage import measure
from scipy import ndimage as ndi
from skimage.feature import canny
from PIL import Image
from sklearn.svm.classes import SVC
import cPickle
import time
caffe_path = '/home/harp/SaveData/Sharon/caffe/python'
sys.path.insert(0,caffe_path)
import caffe

#parameters
#image path
img_path = sys.argv[1]
gt_path = sys.argv[2]
#number of pixels to vote for classification in each superpixel
max_samples = 19
#path to load random forest model
rf_model = sys.argv[3] 
#FCN model
CAFFEMODEL = sys.argv[4]

caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Net('deploy.prototxt', CAFFEMODEL, caffe.TEST)
#classification result path
output_segmentmap_path = os.path.dirname(img_path)+"/"+os.path.basename(img_path)[:os.path.basename(img_path).rfind(".")]+"_cls.jpg"
output_mix_path = os.path.dirname(img_path)+"/"+os.path.basename(img_path)[:os.path.basename(img_path).rfind(".")]+"_mix.jpg"
output_superpixel_path = os.path.dirname(img_path)+"/"+os.path.basename(img_path)[:os.path.basename(img_path).rfind(".")]+"_sp.jpg"

knownLabels = ['background']+open('classes_APC.txt').readlines()
numKnownLabels = len(knownLabels)
unknownLabels = open('classes_unknown.txt').readlines()
numUnknownLabels = len(unknownLabels)
totalLabels = numKnownLabels + numUnknownLabels
def superpixel_segmentation():
	segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
	segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
	segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
	gradient = sobel(rgb2gray(img))
	segments_watershed = watershed(gradient, markers=250, compactness=0.001)

	print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
	print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
	print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

	fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True,
	                       subplot_kw={'adjustable': 'box-forced'})

	ax[0, 0].imshow(mark_boundaries(img, segments_fz))
	ax[0, 0].set_title("Felzenszwalbs's method")
	ax[0, 1].imshow(mark_boundaries(img, segments_slic))
	ax[0, 1].set_title('SLIC')
	ax[1, 0].imshow(mark_boundaries(img, segments_quick))
	ax[1, 0].set_title('Quickshift')
	ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
	ax[1, 1].set_title('Compact watershed')

	for a in ax.ravel():
	    a.set_axis_off()

	plt.tight_layout()
	plt.savefig(output_superpixel_path)
	plt.close()

	return segments_watershed


def feature_layers_extact(img_path):
	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	im = Image.open(img_path)
	# im.thumbnail(size)
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))
	in_ = in_.transpose((2,0,1))

	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_

	# run net and take argmax for prediction
	net.forward()

	imgsize = im.size[0],im.size[1]  # same as input image size (640,480)
	print "image size",imgsize
	feats = []
	scales = []
	print("Extracting FCN feature layers...")

	for feat_name in feature_layers:
		#print_shape(feat_name)
		feat = net.blobs[feat_name].data[0]
		feats.append(feat)
		scales.append([feat.shape[2]/float(imgsize[0]), feat.shape[1]/float(imgsize[1])])

	return feats, scales

def fcn_result(gt_path):
	#gt_path = gt_path + os.path.basename(img_path)[:os.path.basename(img_path).rfind(".")] + '.png'
	#im_cls = np.asarray(Image.open(gt_path))
	#classes = np.unique(im_cls)
	classes = [0, 5, 31, 34]
	#print ("classes: {}".format(classes))

	label_zero_out = list(set(range(numKnownLabels))-set(classes))
	outs = net.blobs['score'].data[0]
	outs[label_zero_out, :, :] = 0
	out = outs.argmax(axis=0)
	scores = np.unique(out)
	labels = [knownLabels[s] for s in scores]
	plt.imshow(out,vmin =0, vmax = numKnownLabels, cmap='hsv',interpolation='none')
	formatter = plt.FuncFormatter(lambda val, loc: knownLabels[val])
	plt.colorbar(ticks=range(0, len(knownLabels)), format=formatter)
	plt.clim(-0.5, len(knownLabels) - 0.5)
	plt.imsave(img_path[:img_path.rfind(".")]+'_fcn.jpg', np.uint8(out), vmin=0, vmax = totalLabels, cmap='hsv')
	
	return out
#main function
if __name__ == '__main__':
	start = time.time()

	#superpixel segmentation
	img = io.imread(img_path) #img = gray2rgb(img)

	feature_layers = ['conv3_3', 'conv4_3', 'conv5_3', 'fc7']
	feats, scales = feature_layers_extact(img_path)

	#load SVM model
	print "Load SVM model for prediction..."
	rf = cPickle.load(open(rf_model,"rb"))

	print "Loop through superpixels..."
	#region classification
        segmentation = superpixel_segmentation()
	regions = measure.regionprops(segmentation, intensity_image=rgb2gray(img))
	#ind_classes = []

	print "Number of regions: ", len(regions)
	rcount = 0
	result = np.zeros((img.shape[0], img.shape[1]), dtype=np.int8)
	for r in regions:
		mask = r.convex_image
		bbx  = r.bbox
		tars = np.where(mask == 1)
		ind  = [tars[0]+bbx[0],tars[1]+bbx[1]]   # pixel locations

		p = np.random.permutation(len(ind[0]))
		num_pixel = min(max_samples, len(ind[0]))
		ind_shuffled = [ind[0][p][:num_pixel], ind[1][p][:num_pixel]]
		#ind_classes.append(ind_shuffled)

		pixel_feats = np.empty((0, 5376), float)
		for x in range(len(ind_shuffled[0])):
			i = ind_shuffled[0][x]
			j = ind_shuffled[1][x]
			pixel_feat = np.empty((0), float)
			for k in range(len(feature_layers)):
				feat_vector = feats[k][:,int(i*scales[k][1]), int(j*scales[k][0])]
				pixel_feat = np.hstack((pixel_feat, feat_vector))
			pixel_feat=pixel_feat.reshape((1,-1))
			#print "pixel_feat shape: ", pixel_feat.shape
			pixel_feats = np.concatenate((pixel_feats, pixel_feat),axis=0)

		#print "pixel_feats shape: ", pixel_feats.shape
		#labels_prob = rf.predict_proba(pixel_feats)
		labels_pred = rf.predict(pixel_feats)

		labels_pred = labels_pred.astype(int)
		label = np.argmax(np.bincount(labels_pred), axis = 0)
		#print "preds : ", labels_pred
		#print "label : ", label

		rcount += 1

		#visualize the superpixel segmentation result
		#img_trans = img.copy()
		#img_trans[ind_shuffled] = (255,0,0)
		#plt.imshow(img_trans)
		#plt.savefig('segmentation_region')
		#a = raw_input('Next plot? y/n\n')
		#plt.clf()
		#np.savetxt(fout_feat, pixel_feat.reshape(1,-1), fmt='%.7f', delimiter=',')
		if label != 0:
			result[ind] = label + numUnknownLabels

	#plt.imshow(result, vmin =0, vmax = 41, cmap='hsv',interpolation='none')
	#plt.savefig(output_segmentmap_path)
        plt.imsave(output_segmentmap_path, result, vmin=0, vmax = totalLabels, cmap='hsv')
	fcn_out = fcn_result(gt_path)
	#try to overwrite background result in fcn_out
	idx = np.where((fcn_out == 0) & (result != 0))
	fcn_out[idx] = result[idx]
        plt.imsave(output_mix_path, fcn_out, vmin=0, vmax = totalLabels, cmap='hsv')
	plt.close()
	end = time.time()
	print "time: ", end-start, "s"





