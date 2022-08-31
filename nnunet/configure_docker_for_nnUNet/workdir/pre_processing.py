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

def load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing):
    img = imread(img_file)
    img_itk = sitk.GetImageFromArray(img.astype(np.float32))
    img_itk.SetSpacing(np.array(spacing)[::-1])
    sitk.WriteImage(img_itk, join(img_out_base + "_0000.nii.gz"))

    if lab_file is not None:
        l = imread(lab_file)
        l = (l / 255).astype(int) # set label to 0 or 1
        l[l > 0] = 1
        l_itk = sitk.GetImageFromArray(l.astype(np.uint8))
        l_itk.SetSpacing(np.array(spacing)[::-1])
        sitk.WriteImage(l_itk, join(anno_out+'.nii.gz'))


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

def add_zeros_before(i):
    # converts i to '00i'
    if i < 10:
        return '00{}'.format(i)
    elif i < 100:
        return '0{}'.format(i)
    else:
        return '{}'.format(i)


# converts all images to nifti format
import tqdm
list_imgs = os.listdir(img_folder)
for i in range(len(list_imgs)):
#     img = imread(img_folder + list_imgs[i])
#     msk = imread(msk_folder + list_imgs[i])
    
    # input name
    base_name = list_imgs[i][:str.rfind(list_imgs[i], '.')]
    
    img_out_base = img_out_base_folder+case+'_'+add_zeros_before(i)
    anno_out = anno_out_folder+case+'_'+add_zeros_before(i)
    
    
    load_tiff_convert_to_nifti(
        img_folder + '/' + list_imgs[i], 
        msk_folder + '/' + list_imgs[i], 
        img_out_base, 
        anno_out, 
        spacing)


# move test images to the right folder
def get_train_test_df(df):
    """
    Return the train set and the test set
    """
    train_set = np.array(df4[df4['fold_x']==0].iloc[:,0])
    test_set = np.array(df4[df4['fold_x']==1].iloc[:,0])
    return train_set, test_set

#Le fichier 'folds_x_sophie.csv' doit etre dans le répertoire courant
df4 = pd.read_csv('folds_x_sophie.csv')

train_set, test_set = get_train_test_df(df4)
print("Size of train set {}".format(len(train_set)))
print("Size of test set {}".format(len(test_set)))

# copy test files to test folder
# removes them from the train folder
list_imgs = os.listdir(img_folder)
for i in range(len(list_imgs)):
    base_name = list_imgs[i]
    if base_name in test_set:
        img_out_base = case+'_'+add_zeros_before(i)+'_0000.nii.gz'
        if os.path.exists(img_out_base_folder+img_out_base):
            copyfile(img_out_base_folder+img_out_base, img_test_out_base_folder+img_out_base)
            os.remove(img_out_base_folder+img_out_base)
            
        anno_out_base = case+'_'+add_zeros_before(i)+'.nii.gz'
        if os.path.exists(anno_out_folder+anno_out_base):
            copyfile(anno_out_folder+anno_out_base, anno_test_out_folder+anno_out_base)
            os.remove(anno_out_folder+anno_out_base)



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