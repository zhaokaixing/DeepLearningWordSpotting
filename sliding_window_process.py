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
from detect_word_region import DetectRegion
from scipy.misc import imresize
from generate_caffe_class import CaffeFeatureGeneration
from record import Record

class SlidingWindow(object):

	def generate_sliding_window_with_text_region_detection(self, image, window_x_start, window_x_end, window_y_start, window_y_end, stepSize_x, stepSize_y,  windowSize):
	    for y in range(window_y_start, window_y_end, stepSize_y):
		for x in range(window_x_start, window_x_end, stepSize_x):
		    yield(x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

	def generate_sliding_window_without_text_region_detection(self, image, stepSize_x, stepSize_y, windowSize):
	    for y in range(0, image.shape[0], stepSize_y):
		for x in range(0, image.shape[1], stepSize_x):
		    yield(x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

	def process_sliding_window(self, flag, image_path, window_image_path, result_line_path):
		print('ok')
		image = cv2.imread(image_path)
		sp = image.shape
		sz1 = sp[0]
		sz2 = sp[1]
		(winW, winH) = (45, 55)
		i = 0
		sub_window = []
		generate_class = CaffeFeatureGeneration()
		object_array = []
		if flag == 0:
		    for (x, y, window) in self.generate_sliding_window_without_text_region_detection(image, stepSize_x=2, stepSize_y=60, windowSize=(winW, winH)):
			    if x + winW > sz2:
				break
			    if y + winH > sz1:
				break
			    sub_window_coord = []
			    sub_window_coord.append(x)
			    sub_window_coord.append(y)
			    sub_window_coord.append(winW)
			    sub_window_coord.append(winH)
			    new_image = image.copy()
			    cv2.rectangle(new_image, (x, y), (x + winW, y + winH), (0, 255, 0), 5)
			    win = cv2.namedWindow('Window', flags=0)
			    cv2.imshow('Window', new_image)
			    cv2.waitKey(1)
			    #time.sleep(1)
			    cv2.imwrite(window_image_path + str(i) + '.png', window)
			    path_window = window_image_path + str(i) + '.png'
			    i = i + 1
			    class_num = generate_class.generate_feature(path_window)
			    sub_window_coord.append(class_num)
			    record_object = Record(x, y, class_num)
			    object_array.append(record_object)
			    f = open(result_line_path, 'a')
			    f.write(str(sub_window_coord)+'\n')
		elif flag == 1:
		    detect_region = DetectRegion()
	    	    box = detect_region.detect(image)
		    for (x, y, window) in self.generate_sliding_window_with_text_region_detection(image, window_x_start=box[0][0], window_x_end=box[2][0], window_y_start=box[1][1], window_y_end=box[3][1], stepSize_x=winW, stepSize_y=85, windowSize=(winW, winH)):
			    print('flag==1')
			    if x + winW > sz2:
				break
			    if y + winH > sz1:
				break
			    sub_window_coord = []
			    sub_window_coord.append(x)
			    sub_window_coord.append(y)
			    sub_window_coord.append(winW)
			    sub_window_coord.append(winH)
			    new_image = image.copy()
			    cv2.rectangle(new_image, (x, y), (x + winW, y + winH), (0, 255, 0), 5)
			    win = cv2.namedWindow('Window', flags=0)
			    cv2.imshow('Window', new_image)
			    cv2.waitKey(1)
			    #time.sleep(1)
			    cv2.imwrite(window_image_path + str(i) + '.png', window)
			    path_window = window_image_path + str(i) + '.png'
			    i = i + 1
			    class_num = generate_class.generate_feature(path_window)
			    sub_window_coord.append(class_num)
			    record_object = Record(x, y, class_num)
			    object_array.append(record_object)
			    f = open(result_line_path, 'a')
			    f.write(str(sub_window_coord)+'\n')

		return object_array
