import os
import shutil
import argparse
import nibabel as nib
import pandas as pd
from os.path import join
import numpy as np
from skimage.io import imread
import SimpleITK as sitk
import matplotlib.pyplot as plt
from shutil import copyfile
from nnunet.utils import generate_dataset_json

parser = argparse.ArgumentParser(description="Main training file.")
parser.add_argument("-i", "--images_path", type=str, default="single",
    help="The past to the raw data (images)")
parser.add_argument("-l", "--labels_path", type=str,
    help="The past to the raw data (labels)")
parser.add_argument("-t", "--task_name", type=str,
    help="The task name")

args = parser.parse_args()
args = vars(args)

images_path = args["images_path"]
labels_path = args["labels_path"]
task_name   = args["task_name"]



#Attention : Les dossiers 'images_sophie', 'masks_sophie' doivent etre dans le répertoire courant
idx = 'M1_2'
postfix = '_C0.tif'
base_name = 'KAKU4-wt--CRWN1-wt--CRWN4-wt_Cot_J13_STD_FIXE_H258_{}'.format(idx)

img_folder = images_path
img_file = img_folder+base_name+postfix

msk_folder = labels_path
msk_file = msk_folder+base_name+postfix

img_out_base_folder      = f"nnUNet_raw_data_base/nnUNet_raw_data/Task500_{task_name}/imagesTr/"
img_test_out_base_folder = f"nnUNet_raw_data_base/nnUNet_raw_data/Task500_{task_name}/imagesTs/"
img_out_base = img_out_base_folder+base_name

anno_out_folder = f"nnUNet_raw_data_base/nnUNet_raw_data/Task500_{task_name}/labelsTr/"
anno_test_out_folder = f"nnUNet_raw_data_base/nnUNet_raw_data/Task500_{task_name}/labelsTs/"
anno_out = anno_out_folder+base_name

case = task_name

target_base = f"nnUNet_raw_data_base/nnUNet_raw_data/Task500_{task_name}"

spacing=(0.2e-3, 0.1032e-3,0.1032e-3)
print("Problems")
print(join(target_base,'dataset.json'))
print(img_out_base_folder)
print(img_test_out_base_folder)
print(case)
print("Problems")

# dataset.json generator
generate_dataset_json(
    join(target_base,'dataset.json'),
    img_out_base_folder,
    img_test_out_base_folder,
    modalities=('D'),
    labels = {0: 'background', 1: task_name},
    dataset_name=case,
    license='MIT'
)