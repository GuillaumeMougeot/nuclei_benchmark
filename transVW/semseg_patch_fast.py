#---------------------------------------------------------------------------
# Dataset primitives for 3D segmentation dataset
# solution: patch approach with the whole dataset into memory 
#---------------------------------------------------------------------------

import os
import numpy as np 
import torchio as tio
import random 
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from tqdm import tqdm 
from skimage.io import imread

import utils 
import nibabel as nib

#---------------------------------------------------------------------------
# utilities to random crops
    
# create a patch centered on this voxel
def centered_crop(img, msk, center, crop_shape):
    """
    centered crop a portion of size prop of the original image size.
    """
    crop_shape=np.array(crop_shape)
    start = center-crop_shape//2
    end = crop_shape+start

    crop_img = img[:,
                start[0]:end[0], 
                start[1]:end[1], 
                start[2]:end[2]]
    crop_msk = msk[:,
                start[0]:end[0], 
                start[1]:end[1], 
                start[2]:end[2]]
    return crop_img, crop_msk

def random_crop(img, msk, crop_shape):
    """
    randomly crop a portion of size prop of the original image size.
    """
    img_shape = np.array(img.shape)[1:]
    rand_start = np.array([random.randint(0,c) for c in np.maximum(0,(img_shape-crop_shape))])
    rand_end = crop_shape+rand_start

    crop_img = img[:,
                    rand_start[0]:rand_end[0], 
                    rand_start[1]:rand_end[1], 
                    rand_start[2]:rand_end[2]]
    crop_msk = msk[:,
                    rand_start[0]:rand_end[0], 
                    rand_start[1]:rand_end[1], 
                    rand_start[2]:rand_end[2]]
    return crop_img, crop_msk

def pad(img, final_size):
    """
    randomly pad an image with zeros to reach the final size. 
    if the image is bigger than the expected size, then the image is cropped.
    """
    img_shape = np.array(img.shape)[1:]
    size_range = (final_size-img_shape) * (final_size-img_shape > 0) # needed if the original image is bigger than the final one
    rand_start = np.array([random.randint(0,c) for c in size_range])

    rand_end = final_size-(img_shape+rand_start)
    rand_end = rand_end * (rand_end > 0)

    pad = np.append([[0,0]],np.vstack((rand_start, rand_end)).T,axis=0)
    pad_img = np.pad(img, pad, 'constant', constant_values=0)

    # crop the image if needed
    if ((final_size-img_shape) < 0).any(): # keeps only the negative values
        pad_img = pad_img[:,:final_size[0],:final_size[1],:final_size[2]]
    return pad_img

def random_crop_pad(img, msk, final_size, fg_rate=0.33):
    """
    random crop and pad if needed.
    """
    # choose if using foreground centrered or random alignement
    force_fg = random.random()
    if fg_rate>0 and force_fg<fg_rate:
        rnd_label = random.randint(0,msk.shape[0]-1) # choose a random label
        center=random.choice(np.array(np.where(msk[rnd_label] == 1)).T) # choose a random voxel of this label
        img, msk = centered_crop(img, msk, center, final_size)
    else:
        # or random crop
        img, msk = random_crop(img, msk, final_size)

    # pad if needed
    if (np.array(img.shape)[1:]-final_size).sum()!=0:
        img = pad(img, final_size)
        msk = pad(msk, final_size)
    return img, msk

#---------------------------------------------------------------------------

class SemSeg3DPatchFast(Dataset):
    """
    with DataLoader
    """
    def __init__(
        self,
        img_dir,
        msk_dir,
        batch_size, 
        patch_size,
        nbof_steps,
        folds_csv  = None, 
        fold       = 0, 
        val_split  = 0.25,
        train      = True,
        use_aug    = True,
        fg_rate = 0.33, # if > 0, force the use of foreground, needs to run some pre-computations
        ):

        self.img_dir = img_dir
        self.msk_dir = msk_dir

        self.batch_size = batch_size
        self.patch_size = patch_size

        self.nbof_steps = nbof_steps
        
        # get the training and validation names 
        if folds_csv is not None:
            df = pd.read_csv(folds_csv)
            trainset, testset = utils.get_folds_train_test_df(df, verbose=False)

            self.fold = fold
            
            self.val_imgs = trainset[self.fold]
            del trainset[self.fold]
            self.train_imgs = []
            for i in trainset: self.train_imgs += i

        else: # tmp: validation split = 50% by default
            all_set = os.listdir(img_dir)
            val_split = np.round(val_split * len(all_set)).astype(int)
            val_split = 0
            #if val_split == 0: val_split=1
            self.train_imgs = all_set[val_split:]
            self.val_imgs = all_set[:val_split]
            testset = []
            
        self.train = train
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        length of the testing set: {}\n \
        is training mode active?: {}".format(fold, len(self.train_imgs), len(self.val_imgs), len(testset), self.train))

        self.use_aug = use_aug

        # data augmentation
        ps = np.array(self.patch_size)

        # [aug] 'axes' for tio.RandomAnisotropy
        anisotropy_axes=tuple(np.arange(3)[ps/ps.min()>3].tolist())

        # [aug] 'degrees' for tio.RandomAffine
        norm = ps*3/ps.min()
        softmax=np.exp(norm)/sum(np.exp(norm))
        degrees=softmax.min()*90/softmax

        self.transform = tio.Compose(([
            # spatial augmentations
            tio.RandomFlip(p=0.5, axes=(0,1,2)),
            tio.RandomAnisotropy(p=0.25, axes=anisotropy_axes, downsampling=(1,2)),
            tio.RandomAffine(p=0.2, scales=(0.7,1.4), degrees=degrees, translation=0),
            # tio.RandomElasticDeformation(p=0.2, num_control_points=4, locked_borders=1),
            # tio.OneOf({
            #     tio.RandomAffine(scales=0.1, degrees=10, translation=0): 0.8,
            #     tio.RandomElasticDeformation(): 0.2,
            # }),
            

            # intensity augmentations
            # tio.RandomMotion(p=0.2),
            # tio.RandomGhosting(p=0.2),
            # tio.RandomSpike(p=0.2),
            tio.RandomBiasField(p=0.15, coefficients=0.2),
            tio.RandomBlur(p=0.2, std=(0.5,1.5)),
            # tio.RandomNoise(p=0.2),
            tio.RandomSwap(p=0.2, patch_size=ps//8),
            tio.RandomGamma(p=0.15),
        ] if self.use_aug else []))

        self.fg_rate = fg_rate if self.train else 1
    
    def __len__(self):
        # return len(self.train_imgs) if self.training else len(self.val_imgs)
        return self.nbof_steps*self.batch_size
    
    def __getitem__(self, idx):
        fnames = self.train_imgs if self.train else self.val_imgs
        # img_fname = np.random.choice(fnames)
        img_fname = fnames[idx%len(fnames)]

        # file names
        img_path = os.path.join(self.img_dir, img_fname)
        msk_path = os.path.join(self.msk_dir, img_fname)

        # read the images  
        img = nib.load(img_path)
        img = np.array(img.dataobj)
        img = np.expand_dims(img, axis=0)
        msk = nib.load(msk_path)
        msk = np.array(msk.dataobj)
        msk = np.expand_dims(msk, axis=0)

        # random crop and pad
        img, msk = random_crop_pad(img, msk, final_size=self.patch_size, fg_rate=self.fg_rate)

        # to tensor
        img = torch.from_numpy(img).float()
        msk = torch.from_numpy(msk).float()

        # data augmentation
        if self.train and self.use_aug:
            sub = tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk))
            sub = self.transform(sub)
            img, msk = sub.img.tensor, sub.msk.tensor
        
        return img, msk

#---------------------------------------------------------------------------