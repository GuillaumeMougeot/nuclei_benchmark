import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from cellpose import utils, io, models
import tifffile as tiff
import argparse


parser = argparse.ArgumentParser(description="Prediction files.")
parser.add_argument("-i", "--in_dir", type=str, default="dataset/in",
	help="Input images folder")
parser.add_argument("-o", "--out_dir", type=str, default="dataset/out",
	help="Output images folder")
args = parser.parse_args()


images_path = args.in_dir
result_path = args.out_dir
all_images = os.listdir(images_path)


model = models.Cellpose(gpu=False, model_type='cyto2')

channels = [0,0]

for file in all_images:
	img = io.imread(images_path+'/'+file)
	masks, flows, styles, diams = model.eval(img, do_3D=True, channels=channels)

	io.imsave(result_path+'/'+file, masks)
