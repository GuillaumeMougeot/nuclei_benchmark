from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
#matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

from stardist import random_label_cmap
from stardist.models import StarDist3D

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

import argparse
import shutil
import os


axis_norm = (0,1)

def save_image_tiff(img, labels, path):
    save_tiff_imagej_compatible(path, labels, axes='ZYX')

def normalize_image(img):
    return normalize(img, 1,99.8, axis=axis_norm)



def main():
	parser = argparse.ArgumentParser(description="Prediction files.")
	parser.add_argument("-i", "--in_dir", type=str, default="/stardist/dataset/in",
        	help="Input images folder")
	parser.add_argument("-o", "--out_dir", type=str, default="/stardist/dataset/out",
	        help="Output images folder")
	parser.add_argument("-m", "--model_dir", type=str, default="/stardist/dataset/model",
       	        help="Pretrained model name")
	parser.add_argument("-n", "--model_name", type=str, default="stardist",
        	help="Pretrained model name")
	args = parser.parse_args()


	input_dir = args.in_dir
	output_dir = args.out_dir
	model_name = args.model_name
	model_dir = args.model_dir

	X1 = sorted(glob(input_dir+"/*"))
	X = list(map(imread,X1))

	if (X == []):
		print("Dataset is empty")
		exit()



	for file in glob(os.path.join(input_dir, "*.tif")):
		shutil.copy(file, output_dir)

	n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
   # normalize channels independently
# axis_norm = (0,1,2,3) # normalize channels jointly
	if n_channel > 1:
        	print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

	model = StarDist3D(None, name=model_name, basedir=model_dir)


	H1 = sorted(glob(output_dir+"/*"))
	H = list(map(imread,H1))

	for name,i in zip(H1,H):
    		img = normalize_image(i)
    		labels, details = model.predict_instances(img)
    		save_image_tiff(img, labels, name)



main()
