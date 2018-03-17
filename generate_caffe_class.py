import sys
caffe_root = '/home/zkx/caffe/'
sys.path.insert(0, caffe_root + 'python')
import sys
import caffe
import cv2
import Image
import numpy as np
import time
import os
from scipy.misc import imresize
from generate_feature import FeatureGeneration

class CaffeFeatureGeneration(FeatureGeneration):
	def generate_feature(self, image_path):
		caffe_model = caffe_root + 'prd_un_cnn_13/_iter_10000.caffemodel'
		net_file = caffe_root + 'prd_un_cnn_13/deploy.prototxt'
		mean_file=caffe_root + 'prd_un_cnn_13/mean.binaryproto'
		print('Params loaded!')

		a=caffe.io.caffe_pb2.BlobProto()
		file=open(mean_file,'rb')
		data = file.read()
		a.ParseFromString(data)
		means=a.data
		means=np.asarray(means)
		means=means.reshape(3,55,55)

		net = caffe.Net(net_file,caffe_model,caffe.TEST)
		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
		transformer.set_mean('data', means)
		transformer.set_raw_scale('data', 255) 
		transformer.set_channel_swap('data', (2,1,0))

		im=caffe.io.load_image(image_path)
		net.blobs['data'].data[...] = transformer.preprocess('data',im)
		out = net.forward()
		prob= net.blobs['ip2'].data[0].flatten()
		order=prob.argsort()[-1]
		print(out)
		return order
