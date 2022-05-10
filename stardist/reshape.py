import numpy as np
from glob import glob
from tqdm import tqdm
from tifffile import imread
from tifffile import imsave

import torchio as tio
import pandas as pd
import os
import statistics



####################################################
# Takes a list as param (list_shape)
#
# Returns the median shape of all the list of shapes
####################################################
def compute_median(list_shape):
	m = np.array(list_shape)
	m = np.median(m, axis=0)
	p = 2**5 * (m/2**5).astype(int)
	return p



####################################################################################################
# Takes an image and a median shape
# If the image shape is lower than the median shape it applies pad until it reaches the median shape
# Returns the image
####################################################################################################
def pad_img(img, med):
	shape_img = img.shape
	print(shape_img)
	n_all = [0,0,0]
	for i in range(len(med)):
		if shape_img[i] < med[i]:
			n_all[i] = med[i] - shape_img[i]
	np.pad(img, [(0, n_all[0]), (0, n_all[1]), (0, n_all[2])])
	print(img.shape)





path_img = "/home/mezquitap/nuclei_benchmark/stardist/dataset/in/"

imgs = os.listdir(path_img)

all_list = []
for i in imgs:
	all_list.append(path_img+i)

X_trn = list(map(imread,all_list))

all_shapes = []
for i in X_trn:
	all_shapes.append(list(i.shape))

p = compute_median(all_shapes)
print(p)

for i in range(len(X_trn)):
	print(i)
	pad_img(X_trn[i], p)
	input()
