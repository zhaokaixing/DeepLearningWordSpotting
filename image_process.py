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
from generate_caffe_class import CaffeFeatureGeneration
from sliding_window_process import SlidingWindow

class InputImageProcessing(object):
	def process_image(self, flag, img_path, window_img_path, record_txt_path):
		generate_window_object = SlidingWindow()
		record_object_array = generate_window_object.process_sliding_window(flag, img_path, window_img_path, record_txt_path)
		return record_object_array

