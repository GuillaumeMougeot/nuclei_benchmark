import os
import argparse
import nibabel as nib
import pandas as pd
from os.path import join
import numpy as np
from skimage.io import imread
import SimpleITK as sitk
import matplotlib.pyplot as plt
from shutil import copyfile
#from nnunet.utils import generate_dataset_json

parser = argparse.ArgumentParser(description="Main training file.")
parser.add_argument("-i", "--input_path", type=str, default="single",
    help="The past to the input directory")
parser.add_argument("-o", "--output_path", type=str,
    help="The past to the output directory")

args = parser.parse_args()
args = vars(args)

input  = args["input_path"]
output = args["output_path"]

def nii2np_single(img_path):
    """
    convert nifti format (.nii.gz) to numpy array
    """
    img = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img)
    return img_np

def nii2tif_single(nii_path, tif_path, resample):
    """
    load a nifti file and save it into a tif
    """
    img = nii2np_single(nii_path)
    img = resample(img)
    img = img.astype('float32')
    print(img.dtype)
    io.imsave(tif_path, img)

def abs_path(root, listdir_):
    """
    absolute path
    add root to the beginning of each path in listdir
    """
    listdir = listdir_.copy()
    for i in range(len(listdir)):
        listdir[i] = root + '/' + listdir[i]
    return listdir

def abs_listdir(path):
    """
    absolute path
    read all the path of files stored in 'path' 
    and add root to the beginning of each path in listdir
    """
    return abs_path(path, os.listdir(path))

def nii2tif_folder(nii_folder, tif_folder, resample):
    """
    load a folder of nifti file and save it into a folder of tif
    """
    list_rel = os.listdir(nii_folder)
    list_abs = abs_listdir(nii_folder)
    for i, nii_path in enumerate(list_abs):
        print('Loading index: {:d}/{}'.format(i+1, len(list_abs)), end='')
        print('{:s}\r'.format(''), end='', flush=True)
        
        end = list_rel[i][list_rel[i].rfind('.'):]
        if end=='.gz': # assert it is a nifti file
            tif_path = list_rel[i][:list_rel[i].rfind('.')]
            tif_path = tif_path[:tif_path.rfind('.')]
            tif_path = os.path.join(tif_folder, tif_path+'.tif')
            nii2tif_single(nii_path, tif_path, resample)

import torchio as tio
def resample(img, size=(128,128,128), rerange_image=False, rerange_label=False):
    transform = tio.transforms.Resize(target_shape=size)
    img_tmp = np.expand_dims(img,0)
    # for label: rerange 
#     print(np.max(img_tmp))
    img_tmp = transform(img_tmp)
    if rerange_label:
#         print(type(img_tmp[0][0][0][0]))
        img_tmp = (img_tmp > 0).astype(np.uint8)*255
        img_tmp = img_tmp.astype(np.uint8)
        if len(np.unique(img_tmp[0]))!=2:
            print('error')
    elif rerange_image:
        img_tmp = (img_tmp - img_tmp.min()) / (img_tmp.max() - img_tmp.min())
    return img_tmp[0]


from skimage import io
nii2tif_folder(input, output, resample=lambda x: resample(x))