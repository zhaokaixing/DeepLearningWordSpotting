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

class FeatureGeneration(object):
	def generate_caffe_class(self, image_path):
		return 0
