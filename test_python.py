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
import math
from scipy.misc import imresize
from detect_word_region import DetectRegion
from generate_caffe_class import CaffeFeatureGeneration
from sliding_window_process import SlidingWindow
from record import Record
from generate_feature import FeatureGeneration
from image_process import InputImageProcessing

class TestClass(object):
	def generate_list_object(self, document_path):
		object_array = []
		for line in open(document_path): 
			line = line.strip('\n')
			line = line.strip('[')  
			line = line.strip(']')  
	   		split_result = line.split(', ')
			class_number = split_result[len(split_result)-1]
			x = split_result[0]
			y = split_result[1]
			record_object = Record(x, y, class_number)
			object_array.append(record_object)
		return object_array

	def process(self, req_img_path, doc_img_path):
		request_process_object = InputImageProcessing()
		request_record_object_array = request_process_object.process_image(0, req_img_path, '/home/zkx/caffe/prd_un_cnn_13/result_request_img/', 'request_result.txt')
		document_image_object = InputImageProcessing()
		document_record_object_array = document_image_object.process_image(0, doc_img_path, '/home/zkx/caffe/prd_un_cnn_13/result_img/', 'result.txt')
		return request_record_object_array, document_record_object_array

	def compare_with_index_document(self, doc_img_path):
		document_result_array = self.generate_list_object('result.txt')
		request_result_array = self.generate_list_object('request_result.txt')
		len_doc_res_array = len(document_result_array)
		len_req_res_array = len(request_result_array)
		distance_index = []
		i = 0
		image = cv2.imread(doc_img_path)
		while i < (len_doc_res_array - len_req_res_array):
			temp_array = document_result_array[i:i+len_req_res_array]
			j = 0
			distance = 0
			while j < len_req_res_array:
				temp_distance = (int(temp_array[j].class_number) - int(request_result_array[j].class_number))*(int(temp_array[j].class_number) - int(request_result_array[j].class_number))
				distance = distance + temp_distance
				j = j + 1
			if math.sqrt(distance) < 21:
				distance_array = []
				distance_array.append(distance)
				distance_array.append(int(temp_array[0].x))
				distance_array.append(int(temp_array[0].y))
				distance_index.append(distance_array)
				

			i = i + 1
		distance_index.sort()
		#f_distance = open('distance.txt', 'a')
		#for temp in distance_index:
			#f_distance.write(str(temp) + '\n')
		cv2.rectangle(image, (int(distance_index[0][1]), int(distance_index[0][2])), (int(distance_index[0][1])+120, int(distance_index[0][2])+50), (128, 0, 128), 5)
		
		win = cv2.namedWindow('Window', flags=0)
		cv2.imshow('Window', image)
		cv2.waitKey(0)

	def compare(self, doc_img_path, request_result_array, document_result_array):	
		len_doc_res_array = len(document_result_array)
		len_req_res_array = len(request_result_array)
		distance_index = []
		i = 0
		image = cv2.imread(doc_img_path)
		while i < (len_doc_res_array - len_req_res_array):
			temp_array = document_result_array[i:i+len_req_res_array]
			j = 0
			distance = 0
			while j < len_req_res_array:
				temp_distance = (int(temp_array[j].class_number) - int(request_result_array[j].class_number))*(int(temp_array[j].class_number) - int(request_result_array[j].class_number))
				distance = distance + temp_distance
				j = j + 1
			if math.sqrt(distance) < 23:
				distance_array = []
				distance_array.append(distance)
				distance_array.append(int(temp_array[0].x))
				distance_array.append(int(temp_array[0].y))
				distance_index.append(distance_array)

			i = i + 1
		distance_index.sort()
		#f_distance = open('distance.txt', 'a')
		#for temp in distance_index:
			#f_distance.write(str(temp) + '\n')
		cv2.rectangle(image, (int(distance_index[0][1]), int(distance_index[0][2])), (int(distance_index[0][1])+120, int(distance_index[0][2])+50), (128, 0, 128), 5)

		win = cv2.namedWindow('Window', flags=0)
		cv2.imshow('Window', image)
		cv2.waitKey(0)

if __name__ == '__main__':
	test_object = TestClass()
	#request_record_object_array, document_record_object_array = test_object.process('/home/zkx/caffe/prd_un_cnn_13/271-35-req.png', '/home/zkx/caffe/prd_un_cnn_13/271-35.png')
	#test_object.compare('/home/zkx/caffe/prd_un_cnn_13/271-35.png', request_record_object_array, document_record_object_array)
	test_object.compare_with_index_document('/home/zkx/caffe/prd_un_cnn_13/271-35.png')
	
	
			   	
