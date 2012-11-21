import numpy as np

import cv2
import scipy
from scipy import ndimage

from skimage import exposure
import matplotlib.pyplot as plt

class Enum(set):
	def __getattr__(self, name):
		if name in self:
			return name
		raise AttributeError
		
class SushiClassifer:
	def __init__(self):
		SushiTypes = Enum(["AJI", "AMAEBI", "ANAGO", "HAMACHI",
			   	   "IKA", "IKURA", "MAGURO", "SAKE",
			           "SNAPPER", "TAKO", "TOBIKO", "UNAGI",
			           "UNI"])		
		num_types = 13
		self.histograms_hue = [[] for i in xrange(13)]

	def train(self, path="img/unagi/0.jpg"):
		img = cv2.imread(path)
		#contrast stretching
		p2 = np.percentile(img, 2)
		p98 = np.percentile(img, 98)
		img_normalized = exposure.rescale_intensity(img, in_range=(p2, p98))
	
		img_hsv = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2HSV)
	
		temp_hue = cv2.calcHist([img_hsv], channels=[0], mask=None, histSize=[180], ranges=[0,180])
		temp_hue[50] = 0
		self.histograms_hue[7].append(temp_hue)
	
		fig = plt.figure()
		ax = fig.add_subplot(3,1,1)
		ax.plot(temp_hue)
		plt.show()
	
	def classify(self, path="img/sake/1.jpg"):
		#read in the image
		sushi_original = cv2.imread(path)
		
		#resize the image to 200 columns
		sushi_size = sushi_original.shape
		new_rows = 200
		new_cols = int(sushi_size[1]*200.0/sushi_size[0])
		
		#contrast stretching
		p2 = np.percentile(sushi_original, 2)
		p98 = np.percentile(sushi_original, 98)
		sushi_normalized = exposure.rescale_intensity(sushi_original, in_range=(p2, p98))
		
		#image in different color systems
		sushi_small = cv2.resize(sushi_normalized,(new_cols,new_rows))
		sushi_gray = cv2.cvtColor(sushi_small, cv2.COLOR_BGR2GRAY)
		sushi_hsv = cv2.cvtColor(sushi_small, cv2.COLOR_BGR2HSV)
		sushi_middle = sushi_small[(new_rows/2)-40:(new_rows/2)+40,(new_cols/2)-40:(new_cols/2)+40].copy()
		
		#feature 1: remove parts of the image with hues between 30 and 150 (green, blue, purple) and extract the largest component
		sushi_regions = self.stripNonSushiColors(sushi_small, sushi_hsv, sushi_middle)
		
		#feature 2: edge detection
		#sushi_regions = self.detectEdges(sushi_small, sushi_gray)
		
		cv2.imshow('', sushi_regions)
		cv2.waitKey(0)
	
	def stripNonSushiColors(self, img, img_hsv, img_middle):
		#TODO custom defined bound, may want to get from training
		MIN_HUE = np.array([30,40,40])
		MAX_HUE = np.array([150,256,256])
		
		#regions with non-sushi colors
		hue_regions = cv2.inRange(img_hsv, MIN_HUE, MAX_HUE)
		hue_regions = ndimage.median_filter(hue_regions, 3)
		
		#keep largest region
		labeled_img, num_labels = ndimage.label(~hue_regions)
		component_sizes = ndimage.sum(np.ones_like(num_labels), labeled_img, range(num_labels+1))

		mask_sizes = component_sizes != component_sizes.max()
		pixels_to_remove = mask_sizes[labeled_img]
		labeled_img[pixels_to_remove] = 0
		mask = labeled_img > 0
		
		custom color of non-sushi region (bright green, since sushi uses seaweed, which is a very dark green, almost black)
		screen = np.ones(img_hsv.shape, dtype=np.uint8)
		screen[:,:] = [50, 200, 200]
		screen = cv2.cvtColor(screen, cv2.COLOR_HSV2BGR)
		
		result = cv2.bitwise_and(img, img, mask=mask.astype("uint8"))
		screen_mask = cv2.bitwise_and(screen, screen, mask=(~mask).astype("uint8"))
		result = cv2.add(result, screen_mask)

		return result
	
	def detectEdges(self, img, img_gray):
		final = cv2.Canny(img_gray, 80, 120)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		result = cv2.morphologyEx(final,cv2.MORPH_CLOSE,kernel)

		contours, hier = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			cv2.drawContours(result,[cnt],0,255,-1)
		
		return result
		
model = SushiClassifer()
#model.train()
model.classify()