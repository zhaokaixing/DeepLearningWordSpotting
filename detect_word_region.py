import sys

import cv2
import numpy as np

class DetectRegion(object): 
	def preprocess(self, gray):
	    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
	    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

	    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
	    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

	    dilation = cv2.dilate(binary, element2, iterations = 1)

	    erosion = cv2.erode(dilation, element1, iterations = 1)

	    dilation2 = cv2.dilate(erosion, element2, iterations = 3)

	    #cv2.imwrite("binary.png", binary)
	    #cv2.imwrite("dilation.png", dilation)
	    #cv2.imwrite("erosion.png", erosion)
	    #cv2.imwrite("dilation2.png", dilation2)

	    return dilation2



	def detect(self, img):
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	    dilation = self.preprocess(gray)

	    cnts, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

	    # compute the rotated bounding box of the largest contour
	    rect = cv2.minAreaRect(c)
	    box = np.int0(cv2.cv.BoxPoints(rect))
	    #print(box)
	    
	    
	    # draw a bounding box arounded the detected barcode and display the image
	    #cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
	    #cv2.imshow("Image", img)
	    #cv2.imwrite("contoursImage2.png", img)
	    #cv2.waitKey(0)
	    return box	



'''if __name__ == '__main__':
    img = cv2.imread("2750275.png")
    detect(img)'''
