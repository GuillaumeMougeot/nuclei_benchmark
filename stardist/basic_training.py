from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
#matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from tifffile import imsave
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

np.random.seed(42)
lbl_cmap = random_label_cmap()

import torchio as tio
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import argparse
import statistics





parser = argparse.ArgumentParser(description="Prediction files.")
parser.add_argument("-c", "--csv", type=str, default="/stardist/dataset/folds_x_sophie.csv",
	help="CSV file for dataset structure")
parser.add_argument("-i", "--img", type=str, default="/stardist/dataset/in",
	help="Training img directory")
parser.add_argument("-m", "--msk", type=str, default="/stardist/dataset/masks",
	help="Training mask directory")
parser.add_argument("-o", "--model_dir", type=str, default="/stardist/dataset/model",
	help="Model output folder")
parser.add_argument("-n", "--model_name", type=str, default="stardist_model",
	help="New model name")
args = parser.parse_args()







path_csv = args.csv
#path_img = "/home/mezquitap/stardist_dataset/images_resampled_sophie/"
#path_mask = "/home/mezquitap/stardist_dataset/masks_resampled_sophie/"

path_img = args.img+"/"
path_mask = args.msk+"/"

df = pd.read_csv(path_csv)
df.head()


def get_fold_x(df, fold=0):
    """
    Return the test set, the validation set and the training set
    """
    test_set = np.array(df[df['fold_x']==0].iloc[:,0])
    val_set = np.array(df[df['fold_x']==1].iloc[:,0])
    if fold==0: # return the full train set
        fold_set = np.array(df[df['fold_x']>1].iloc[:,0])
    else:
        fold_set = np.array(df[df['fold_x']==(fold+2)].iloc[:,0])
    return test_set, val_set, fold_set
(test_name, val_name, train_name) = get_fold_x(df, fold=0)

def name_to_path_in_array(array, path):
    new_array = []
    for name in array:
        new_array.append(path + name)
    return new_array
X_trn = name_to_path_in_array(train_name, path_img)
Y_trn = name_to_path_in_array(train_name, path_mask)

X_val = name_to_path_in_array(val_name, path_img)
Y_val = name_to_path_in_array(val_name, path_mask)

test = name_to_path_in_array(test_name, path_img)


X_trn = list(map(imread,X_trn))
Y_trn = list(map(imread,Y_trn))
X_val = list(map(imread,X_val))
Y_val = list(map(imread,Y_val))
n_channel = 1 if X_trn[0].ndim == 3 else X_trn[0].shape[-1]





axis_norm = (0,1,2)   # normalize channels independently
# axis_norm = (0,1,2,3) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
    sys.stdout.flush()

X_trn = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_trn)]
X_val = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_val)]
Y_trn = [fill_label_holes(y) for y in tqdm(Y_trn)]
Y_val = [fill_label_holes(y) for y in tqdm(Y_val)]

assert len(X_trn+X_val) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X_trn)+len(X_val))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
#X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
#X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X_trn + X_val))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

extents = calculate_extents(Y_val)
anisotropy = tuple(np.max(extents) / extents)
print('empirical anisotropy of labeled objects = %s' % str(anisotropy))

# 96 is a good default choice (see 1_data.ipynb)
n_rays = 96

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = True and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

# Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)


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

all_shapes = []
for i in X_trn:
        all_shapes.append(list(i.shape))
for i in X_val:
	all_shapes.append(list(i.shape))

p = compute_median(all_shapes)

for i in range(len(X_trn)):
        pad_img(X_trn[i], p)


print(tuple(p))

p = tuple([int(i) for i in p])

conf = Config3D (
    rays             = rays,
    grid             = grid,
    anisotropy       = (1,1,1),
    use_gpu          = use_gpu,
    n_channel_in     = n_channel,
    # adjust for your data below (make patch size as large as possible)
#    train_patch_size = (32,64,64),
    train_patch_size = p,
    train_batch_size = 2,
)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.8)
    # alternatively, try this:
    # limit_gpu_memory(None, allow_growth=True)

model = StarDist3D(conf, name=args.model_name, basedir=args.model_dir)

model._axes_tile_overlap('ZYX')

median_size = calculate_extents((Y_trn+Y_val), np.median)
fov = np.array(model._axes_tile_overlap('ZYX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

def random_fliprot(img, mask, axis=None):
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y


def transform_img_msk(
    idx,
    imgs,
    img_dir,
    msk_dir,
    transform,
):
    img_fname = imgs[idx]
    img_path = os.path.join(img_dir, img_fname)
    msk_path = os.path.join(msk_dir, img_fname)
    print(img_path)
    # load mask and image
    img = imread(img_path)
    msk = imread(msk_path)
    assert img.shape == msk.shape, '[error] expected img and msk sizes must match, img {}, msk {}'.format(img.shape, msk.shape)

    # change data type from np.uint16 to np.int16
    img = img.astype(np.float32)
    msk = msk.astype(np.float32)

    # crop if training
    # if self.train: # TODO: remove resize during inference
    # img, msk = self.resize(img, msk)

    # add channel
    img = np.expand_dims(img, 0)
    msk = np.expand_dims(msk, 0)

    # transform with TorchIO
    tmp_sub = transform(tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk)))
    img, msk = tmp_sub.img.numpy()[0].astype(np.uint16), tmp_sub.msk.numpy()[0].astype(np.uint16)

    return img, msk


def transform():
	spacing = (1.0, 1.0, 0.5159651630825821)
	spacing = np.flip(spacing) # CAREFUL: spacing must be in ZYX and not in XYZ!

	crop_shape = (128,128,128)

	transform = tio.Compose([
	            tio.Resample(spacing),
	            tio.CropOrPad(crop_shape)
	])


	# define list of images
	#img_dir = '../data/nucleus/images_sophie'
	#msk_dir = '../data/nucleus/masks_sophie'
	img_dir = args.img
	msk_dir = args.msk
	imgs = os.listdir(img_dir)


	for idx in range(len(imgs)):
	    img_resampled, msk_resampled = transform_img_msk(
	        idx=idx,
	        imgs=imgs,
	        img_dir=img_dir,
	        msk_dir=msk_dir,
	        transform=transform,
	    )
	    # save the images
	    imsave(path_img+imgs[idx], img_resampled)
	    imsave(path_mask+imgs[idx], msk_resampled)



#try:
model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs=400)
model.optimize_thresholds(X_val, Y_val)
#except ValueError:
#	print("The dataset is not normalized, the dataset in "+args.img+" and in "+args.msk+" will be overwritten")
#	norm_accept = input("Enter (y) to normalize, enter any other key to ignore :")
#	if norm_accept == "y" or norm_accept == "Y":
#		transform()
#		print("Finished, run this program again to train")
#	else:
#		exit(1)
