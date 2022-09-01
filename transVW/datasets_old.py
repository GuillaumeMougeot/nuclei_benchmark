#---------------------------------------------------------------------------
# Dataset primitives for 3D Nucleus dataset
# trained with data augmentation 
# trained with only a single image
# for speed reasons this images is repeated "batch_size" times
#---------------------------------------------------------------------------
from math import degrees
from turtle import pos
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random 
import os
from skimage.io import imread
from skimage.morphology import erosion, dilation
import utils
import torchio as tio 
import nibabel as nib

#---------------------------------------------------------------------------
# primitive for 3D semantic segmentation
# can be used only with fixed sized image

class SemSeg3D(Dataset):
    def __init__(self, 
                 img_dir,
                 msk_dir, 
                 folds_csv, 
                 fold, 
                 train=True, 
                 use_onehot = True,
                 ):

        super(Dataset, self).__init__()

        self.img_dir = img_dir
        self.msk_dir = msk_dir
        
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
            print("On est ici")
            all_set = os.listdir(img_dir)
            val_split = int(0 * len(all_set))
            if val_split == 0: val_split=1
            self.train_imgs = all_set
            self.val_imgs = all_set[:val_split]
            testset = []
            
        self.train = train
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        length of the testing set: {}\n \
        is training mode active?: {}".format(fold, len(self.train_imgs), len(self.val_imgs), len(testset), self.train))

        self.training_transform = tio.Compose([
            # tio.Resample(spacing),
            # tio.CropOrPad(crop_shape),
            # tio.RandomMotion(p=0.2),
            # tio.RandomBiasField(p=0.3),
            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            # tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            tio.OneOf({
                tio.RandomAffine(scales=0.1, degrees=10, translation=5): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }),
            tio.RandomGamma(),
        ]+([tio.OneHot()] if use_onehot else []))

        # self.training_transform = tio.Compose([
        #     tio.Resample(spacing),
        #     tio.CropOrPad(crop_shape),
        # ])

        self.validation_transform = tio.Compose([
            # tio.Resample(spacing),
            # tio.CropOrPad(crop_shape),
        ]+([tio.OneHot()] if use_onehot else []))
    
    def __len__(self):
        return len(self.train_imgs) if self.train else len(self.val_imgs)
        
    def __getitem__(self, idx):
        img_fname = self.train_imgs[idx] if self.train else self.val_imgs[idx]
        img_path = os.path.join(self.img_dir, img_fname)
        msk_path = os.path.join(self.msk_dir, img_fname)
        
        # load mask and image
        img = nib.load(img_path)
        img = np.array(img.dataobj)
        msk = nib.load(msk_path)
        msk = np.array(msk.dataobj)
        assert img.shape == msk.shape, '[error] expected img and msk sizes must match, img {}, msk {}'.format(img.shape, msk.shape)
        
        # normalize
        img = img / (2**16 - 1)
        # msk = msk / (2**8 - 1)
        msk = msk / (np.max(msk))
        
        # crop if training
        # if self.train: # TODO: remove resize during inference
        # img, msk = self.resize(img, msk)
        
        # add channel
        img = np.expand_dims(img, 0)
        msk = np.expand_dims(msk, 0)
        
        # to tensor
        img = torch.tensor(img, dtype=torch.float32)
        msk = torch.tensor(msk, dtype=torch.float32)
        

        # transform with TorchIO
        transform = self.training_transform if self.train else self.validation_transform
        tmp_sub = transform(tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk)))
        img, msk = tmp_sub.img.tensor, tmp_sub.msk.tensor

        #img = img.type(torch.float16)
        #msk = msk.type(torch.float16)
        return img, msk

#---------------------------------------------------------------------------
# primitive for 3D semantic segmentation
# with 3D patches



class SemSeg3DUniformSampler(object):
    """
    inspired by Torchio UniformSampler
    usage:
    >>> t = UniformSampler(
            patch_size=(32,64,64),
            nbof_patch=5)            
    >>> for img, msk in t(image):
    >>>     print(img) 
    """
    def __init__(self, 
                 patch_size, # size of the random crop
                 nbof_patch, # number of random crop in the image before stopping the iteration
                ):
        self.patch_size = patch_size
        self.nbof_patch = nbof_patch
        self.img = None # image
        self.msk = None # mask
        self.transform = None # for data augmentation
    
    def __iter__(self):
        self.crt_patch = 0
        return self
    
    def __next__(self):
        if self.img is None or self.msk is None:
            print("Error, please call the instance and pass an image and a mask")
            raise StopIteration
        
        if self.crt_patch < self.nbof_patch:
            # crop and pad
            img, msk = self.random_crop_pad(
                im=self.img, 
                ms=self.msk,
                final_size=self.patch_size)
            
            # add channel
            img = np.expand_dims(img, 0)
            msk = np.expand_dims(msk, 0)
            
            # augment if needed
            if self.transform is not None:
                tmp_sub = self.transform(tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk)))
                img, msk = tmp_sub.img.tensor, tmp_sub.msk.tensor
                
            # to tensor
            # img = torch.tensor(img, dtype=torch.float)
            # msk = torch.tensor(msk, dtype=torch.float)
                
            self.crt_patch += 1
            return img, msk
        else:
            raise StopIteration
            
    def __call__(self, img, msk, transform=None):
        self.img = img
        self.msk = msk
        self.transform = transform
        return self
    
    def random_crop(self, im, ms, crop_shape):
        """
        randomly crop a portion of size prop of the original image size.
        """
        im_shape = np.array(im.shape)
        rand_start = np.array([random.randint(0,c) for c in np.maximum(0,(im_shape-crop_shape))])
        rand_end = crop_shape+rand_start

        crop_im = im[rand_start[0]:rand_end[0], 
                     rand_start[1]:rand_end[1], 
                     rand_start[2]:rand_end[2]]
        crop_ms = ms[rand_start[0]:rand_end[0], 
                     rand_start[1]:rand_end[1], 
                     rand_start[2]:rand_end[2]]
        return crop_im, crop_ms
    
    def pad(self, im, final_size):
        """
        randomly pad an image with zeros to reach the final size. 
        if the image is bigger than the expected size, then the image is cropped.
        """
        im_shape = np.array(im.shape)
        size_range = (final_size-im_shape) * (final_size-im_shape > 0) # needed if the original image is bigger than the final one
        rand_start = np.array([random.randint(0,c) for c in size_range])

        rand_end = final_size-(im_shape+rand_start)
        rand_end = rand_end * (rand_end > 0)

        pad = np.vstack((rand_start, rand_end)).T
        pad_im = np.pad(im, pad, 'constant')

        # crop the image if needed
        if ((final_size-im_shape) < 0).any(): # keeps only the negative values
            pad_im = pad_im[:final_size[0],:final_size[1],:final_size[2]]
        return pad_im
    
    def random_crop_pad(self, im, ms, final_size):
        """
        random crop and pad if needed.
        """
        im, ms = self.random_crop(im, ms, final_size)
        if (np.array(im.shape)-final_size).sum()!=0:
            im = self.pad(im, final_size)
            ms = self.pad(ms, final_size)
        return im, ms

class SemSeg3DPatch(Dataset):
    def __init__(self, 
                 img_dir,
                 msk_dir, 
                 folds_csv, 
                 fold       = 0, 
                 training   = True, 
                 use_onehot = False,
                 patch_size = (64,64,32),
                 patch_per_img = 1, # number of patch per image during training before reloading a novel image
                ):

        super(Dataset, self).__init__()

        # image and mask dirs
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        
        # get the training and validation names 
        df = pd.read_csv(folds_csv)
        trainset, testset = utils.get_folds_train_test_df(df, verbose=False)
        
        self.fold = fold
        
        self.val_imgs = trainset[self.fold]
        del trainset[self.fold]
        self.train_imgs = []
        for i in trainset: self.train_imgs += i
        
        # training ?
        self.training = training
        
        # verbose
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        length of the testing set: {}\n \
        is training mode active?: {}".format(fold, len(self.train_imgs), len(self.val_imgs), len(testset), self.training))

        # patch generators
        self.patch_size = np.array(patch_size)
        self.patch_per_img = patch_per_img
        self.crt_idx = 0 # this index represents the image that is loaded for training, it changes every 'patch_par_img' time
        self.wait_idx = patch_per_img
        self.training_generator = SemSeg3DUniformSampler(patch_size=patch_size, nbof_patch=patch_per_img)
        self.validation_generator = tio.data.GridSampler
        
        # data augmentation 
        self.training_transform = tio.Compose([
            # tio.RandomMotion(p=0.2),
            # tio.RandomBiasField(p=0.3),
            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            # tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            tio.RandomAffine(scales=0.1, degrees=10, translation=5),
            # tio.OneOf({
            #     tio.RandomAffine(scales=0.1, degrees=10, translation=5): 0.8,
            #     tio.RandomElasticDeformation(): 0.2,
            # }),
            tio.RandomGamma(),
        ]+([tio.OneHot()] if use_onehot else []))

        self.validation_transform = tio.Compose([
        ]+([tio.OneHot()] if use_onehot else []))
    
    def __len__(self):
        return len(self.train_imgs) if self.training else len(self.val_imgs)
        
    def __getitem__(self, idx):
        # get image index for training
        # the idx index is considered once every 'patch_per_img' only
        # TODO: maybe change that behaviour... 
        if self.training and self.wait_idx < self.patch_per_img-1:
            idx = self.crt_idx
            self.wait_idx += 1
        else:
            self.crt_idx = idx
            self.wait_idx = 0
            
            img_fname = self.train_imgs[idx] if self.training else self.val_imgs[idx]
            img_path = os.path.join(self.img_dir, img_fname)
            msk_path = os.path.join(self.msk_dir, img_fname)

            # load mask and image
            img = imread(img_path)
            msk = imread(msk_path)
            assert img.shape == msk.shape, '[error] expected img and msk sizes must match, img {}, msk {}'.format(img.shape, msk.shape)

            # normalize
            bits = lambda x: ((np.log2(x)>8).astype(int)+(np.log2(x)>16).astype(int)*2+1)*8
            img = img / (2**bits(np.max(img)) - 1)
            msk = msk / (2**bits(np.max(msk)) - 1)
            
            if self.training: 
                self.train_it = iter(self.training_generator(img,msk, transform=self.training_transform))
        
        # load patch
        if self.training:
            return next(self.train_it)
        else:
            img = np.expand_dims(img, 0)
            msk = np.expand_dims(msk, 0)
            
            # to tensor
            img = torch.tensor(img, dtype=torch.float)
            msk = torch.tensor(msk, dtype=torch.float)

            # transform with TorchIO, tmp removed because using a resampled dataset
            sub = self.validation_transform(
                tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk)))
            return self.validation_generator(
                subject=sub, 
                patch_size=self.patch_size,
                patch_overlap=self.patch_size//2,
                )


#---------------------------------------------------------------------------
# Dataset primitives for 3D Nucleus dataset
# trained with data augmentation 
# trained with only a single image
# for speed reasons this images is repeated "batch_size" times

class Nucleus3DSingle(Dataset):
    def __init__(self, 
                 img_dir,
                 msk_dir, 
                 folds_csv, 
                 train_idx=None, # index of the current training image
                 dataset_size=None, 
                 train=True, 
                 spacing=None,
                 use_aug = False,  # use augmentation ?
                 use_onehot = True,
                 crop_shape = (128,128,128)):
        super(Dataset, self).__init__()

        self.img_dir = img_dir
        self.msk_dir = msk_dir
        
        df = pd.read_csv(folds_csv)
        trainset, self.val_imgs, testset = utils.get_splits_train_val_test_overlapping(df)
        
        self.train_imgs = trainset[0] # only portion 0 is considered
        if train_idx == None or dataset_size == None: 
            train=False; train_idx=0
        self.train_idx = train_idx
        self.train_img, self.train_msk = self.prepare_train_img_msk()
        self.dataset_size = dataset_size
            
        self.train = train
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        length of the testing set: {}\n \
        is training mode active?: {}".format(0, len(self.train_imgs), len(self.val_imgs), len(testset), self.train))

        self.use_aug = use_aug
        if self.use_aug:
            self.training_transform = tio.Compose([
                tio.RandomFlip(),
                tio.OneOf({
                    tio.RandomAffine(scales=0.1, degrees=10, translation=5): 0.8,
                    tio.RandomElasticDeformation(): 0.2,
                }),
                tio.RandomGamma(),
            ]+([tio.OneHot()] if use_onehot else []))
        else:
            self.training_transform = tio.Compose([]+([tio.OneHot()] if use_onehot else []))

        self.validation_transform = tio.Compose([]+([tio.OneHot()] if use_onehot else []))

    def prepare_train_img_msk(self):
        img_fname = self.train_imgs[self.train_idx] 
        print("Single image used for training:", img_fname)
        img_path = os.path.join(self.img_dir, img_fname)
        msk_path = os.path.join(self.msk_dir, img_fname)
        
        # load mask and image
        img = imread(img_path)
        msk = imread(msk_path)
        assert img.shape == msk.shape, '[error] expected img and msk sizes must match, img {}, msk {}'.format(img.shape, msk.shape)
        
        # normalize
        img = img / (2**16 - 1)
        msk = msk / (2**8 - 1)
        
        # crop if training
        # if self.train: # TODO: remove resize during inference
        # img, msk = self.resize(img, msk)
        
        # add channel
        img = np.expand_dims(img, 0)
        msk = np.expand_dims(msk, 0)
        
        # to tensor
        img = torch.tensor(img, dtype=torch.float)
        msk = torch.tensor(msk, dtype=torch.float)
        return img, msk 
    
    def __len__(self):
        return self.dataset_size if self.train else len(self.val_imgs) # length of the batch_sizeif training
        
    def __getitem__(self, idx):
        if self.train: 
            img = self.train_img
            msk = self.train_msk 
        else: 
            img_fname = self.train_imgs[idx] if self.train else self.val_imgs[idx]
            img_path = os.path.join(self.img_dir, img_fname)
            msk_path = os.path.join(self.msk_dir, img_fname)
            
            # load mask and image
            img = imread(img_path)
            msk = imread(msk_path)
            assert img.shape == msk.shape, '[error] expected img and msk sizes must match, img {}, msk {}'.format(img.shape, msk.shape)
            
            # normalize
            img = img / (2**16 - 1)
            msk = msk / (2**8 - 1)
            
            # add channel
            img = np.expand_dims(img, 0)
            msk = np.expand_dims(msk, 0)
            
            # to tensor
            img = torch.tensor(img, dtype=torch.float)
            msk = torch.tensor(msk, dtype=torch.float)

        # transform with TorchIO, tmp removed because using a resampled dataset
        transform = self.training_transform if self.train else self.validation_transform
        tmp_sub = transform(tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk)))
        img, msk = tmp_sub.img.tensor, tmp_sub.msk.tensor

        return img, msk

#---------------------------------------------------------------------------
# triplet loss dataset pipeline 
# TODO: simplify the data-augmentation

class Triplet(Dataset):
    def __init__(self, 
                 img_dir,
                 folds_csv, 
                 spacing = None,
                 crop_shape = (128,128,128)):
        super(Dataset, self).__init__()

        self.img_dir = img_dir
        
        df = pd.read_csv(folds_csv)
        # TODO: careful below: its the full training set that is selected and not one folds.
        self.train_imgs, testset = utils.get_train_test_df(df, verbose=False)
            
        print("length of the training set: {}\n \
        length of the testing set: {}".format(len(self.train_imgs),len(testset)))

        self.transform = tio.Compose([
            # tio.Resample(spacing),
            # tio.CropOrPad(crop_shape),
            # tio.RandomMotion(p=0.2),
            # tio.RandomBiasField(p=0.3),
            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            # tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            tio.OneOf({
                tio.RandomAffine(scales=0.2, degrees=90, translation=10): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }),
            # tio.RandomAffine(scales=0.2, degrees=90, translation=10),
            tio.RandomGamma(),
        ])
    
    def crop_image(self, im, prop=2/3):
        """
        randomly crop a portion of size prop of the original image size.
        """
        im_shape = np.array(im.shape)
        crop_size = np.round(prop*im_shape).astype(int)
        rand_start = np.array([random.randint(0,c) for c in im_shape-crop_size])
        rand_end = crop_size+rand_start

        crop_im = im[rand_start[0]:rand_end[0], 
                    rand_start[1]:rand_end[1], 
                    rand_start[2]:rand_end[2]]
        return crop_im

    def pad_image(self, im, final_size=(128,128,128)):
        """
        randomly pad an image with zeros to reach the final size. 
        if the image is bigger than the expected size, then the image is cropped.
        """
        im_shape = np.array(im.shape)
        final_size = np.array(final_size)
        size_range = (final_size-im_shape) * (final_size-im_shape > 0) # needed if the original image is bigger than the final one
        rand_start = np.array([random.randint(0,c) for c in size_range])

        rand_end = final_size-(im_shape+rand_start)
        rand_end = rand_end * (rand_end > 0)

        pad = np.vstack((rand_start, rand_end)).T
        pad_im = np.pad(im, pad, 'constant')

        # crop the image if needed
        if ((final_size-im_shape) < 0).any(): # keeps only the negative values
            pad_im = pad_im[:final_size[0],:final_size[1],:final_size[2]]
        return pad_im
    
    def crop_pad_expand_tensor_image(self, im, crop_prop=2/3, final_size=(128,128,128)):
        """
        crop, pad, expand, tensor an image for training
        """
        crop_im = self.crop_image(im, crop_prop)
        pad_im =  self.pad_image(crop_im, final_size)
        expand_im = np.expand_dims(pad_im, 0)
        tensor_im = torch.tensor(expand_im, dtype=torch.float)
        return tensor_im

    def __len__(self):
        return len(self.train_imgs) 
        
    def __getitem__(self, idx):
        img_fname = self.train_imgs[idx]
        idx_neg = np.random.randint(0,self.__len__())
        while idx_neg==idx: idx_neg = np.random.randint(0,self.__len__())
        neg_fname = self.train_imgs[idx_neg]
        img_path = os.path.join(self.img_dir, img_fname)
        neg_path = os.path.join(self.img_dir, neg_fname)
        
        # load mask and image
        img = imread(img_path)
        neg = imread(neg_path)
        
        # normalize
        img = img / (2**16 - 1)
        neg = neg / (2**16 - 1)

        anc = self.crop_pad_expand_tensor_image(img)
        pos = self.crop_pad_expand_tensor_image(img)
        pos = self.transform(pos)
        neg = self.crop_pad_expand_tensor_image(neg)
        neg = self.transform(neg)

        # add channel
        # img = np.expand_dims(img, 0)
        # neg = np.expand_dims(neg, 0)

        # # to tensor
        # img = torch.tensor(img, dtype=torch.float)
        # neg = torch.tensor(neg, dtype=torch.float)
        
        # # crop if training
        # anchor = self.transform(img)
        # positive = self.transform(img)
        # negative = self.transform(neg)
        
        return anc, pos, neg

#---------------------------------------------------------------------------
# Arcface loss dataset pipeline 
# TODO: simplify the data-augmentation

class ArcFace(Dataset):
    def __init__(self, 
                 img_dir,
                 folds_csv, 
                 spacing = None,
                 crop_shape = (128,128,128)):
        super(Dataset, self).__init__()

        self.img_dir = img_dir
        
        df = pd.read_csv(folds_csv)
        # TODO: careful below: its the full training set that is selected and not one folds.
        self.train_imgs, testset = utils.get_train_test_df(df, verbose=False)
            
        print("length of the training set: {}\n \
        length of the testing set: {}".format(len(self.train_imgs),len(testset)))

        self.transform = tio.Compose([
            tio.RandomFlip(),
            # tio.OneOf({
            #     tio.RandomAffine(scales=0.2, degrees=90, translation=10): 0.8,
            #     tio.RandomElasticDeformation(): 0.2,
            # }),
            tio.RandomAffine(scales=0.2, degrees=90, translation=10),
            tio.RandomGamma(),
        ])
    
    def crop_image(self, im, prop=2/3):
        """
        randomly crop a portion of size prop of the original image size.
        """
        im_shape = np.array(im.shape)
        crop_size = np.round(prop*im_shape).astype(int)
        rand_start = np.array([random.randint(0,c) for c in im_shape-crop_size])
        rand_end = crop_size+rand_start

        crop_im = im[rand_start[0]:rand_end[0], 
                    rand_start[1]:rand_end[1], 
                    rand_start[2]:rand_end[2]]
        return crop_im

    def pad_image(self, im, final_size=(128,128,128)):
        """
        randomly pad an image with zeros to reach the final size. 
        if the image is bigger than the expected size, then the image is cropped.
        """
        im_shape = np.array(im.shape)
        final_size = np.array(final_size)
        size_range = (final_size-im_shape) * (final_size-im_shape > 0) # needed if the original image is bigger than the final one
        rand_start = np.array([random.randint(0,c) for c in size_range])

        rand_end = final_size-(im_shape+rand_start)
        rand_end = rand_end * (rand_end > 0)

        pad = np.vstack((rand_start, rand_end)).T
        pad_im = np.pad(im, pad, 'constant')

        # crop the image if needed
        if ((final_size-im_shape) < 0).any(): # keeps only the negative values
            pad_im = pad_im[:final_size[0],:final_size[1],:final_size[2]]
        return pad_im
    
    def crop_pad_expand_tensor_image(self, im, crop_prop=2/3, final_size=(128,128,128)):
        """
        crop, pad, expand, tensor an image for training
        """
        crop_im = self.crop_image(im, crop_prop)
        pad_im =  self.pad_image(crop_im, final_size)
        expand_im = np.expand_dims(pad_im, 0)
        tensor_im = torch.tensor(expand_im, dtype=torch.float)
        return tensor_im

    def __len__(self):
        return len(self.train_imgs) 
        
    def __getitem__(self, idx):
        img_fname = self.train_imgs[idx]
        img_path = os.path.join(self.img_dir, img_fname)
        
        # load mask and image
        img = imread(img_path)
        
        # normalize
        img = img / (2**16 - 1)

        img = self.crop_pad_expand_tensor_image(img)

        return img, torch.tensor(idx)

#---------------------------------------------------------------------------
# Co-training: return the imag
# TODO: simplify the data-augmentation

class CoTrain(Dataset):
    def __init__(self, 
                 img_dir,
                 msk_dir, 
                 folds_csv, 
                 train_idx=None, # index of the current training image
                 batch_size=None, 
                 train=True, 
                 spacing=None,
                 use_aug = False,  # use augmentation ?
                 use_onehot = True,
                 crop_shape = (128,128,128)):
        super(Dataset, self).__init__()

        self.img_dir = img_dir
        self.msk_dir = msk_dir
        
        df = pd.read_csv(folds_csv)
        trainset, self.val_imgs, testset = utils.get_splits_train_val_test_overlapping(df)
        
        self.train_imgs = trainset[0] # only portion 0 is considered
        if train_idx == None or batch_size == None: 
            train=False; train_idx=0
        self.train_idx = train_idx
        self.train_img, self.train_msk = self.prepare_train_img_msk()
        self.batch_size = batch_size
            
        self.train = train
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        length of the testing set: {}\n \
        is training mode active?: {}".format(0, len(self.train_imgs), len(self.val_imgs), len(testset), self.train))

        self.use_aug = use_aug
        if self.use_aug:
            self.training_transform = tio.Compose([
                tio.RandomFlip(),
                tio.OneOf({
                    tio.RandomAffine(scales=0.1, degrees=10, translation=5): 0.8,
                    tio.RandomElasticDeformation(): 0.2,
                }),
                tio.RandomGamma(),
            ]+([tio.OneHot()] if use_onehot else []))
        else:
            self.training_transform = tio.Compose([]+([tio.OneHot()] if use_onehot else []))

        self.validation_transform = tio.Compose([]+([tio.OneHot()] if use_onehot else []))

    def prepare_train_img_msk(self):
        img_fname = self.train_imgs[self.train_idx] 
        print("Single image used for training:", img_fname)
        img_path = os.path.join(self.img_dir, img_fname)
        msk_path = os.path.join(self.msk_dir, img_fname)
        
        # load mask and image
        img = imread(img_path)
        msk = imread(msk_path)
        assert img.shape == msk.shape, '[error] expected img and msk sizes must match, img {}, msk {}'.format(img.shape, msk.shape)
        
        # normalize
        img = img / (2**16 - 1)
        msk = msk / (2**8 - 1)
        
        # crop if training
        # if self.train: # TODO: remove resize during inference
        # img, msk = self.resize(img, msk)
        
        # add channel
        img = np.expand_dims(img, 0)
        msk = np.expand_dims(msk, 0)
        
        # to tensor
        img = torch.tensor(img, dtype=torch.float)
        msk = torch.tensor(msk, dtype=torch.float)
        return img, msk 
    
    def __len__(self):
        return len(self.train_imgs) if self.train else len(self.val_imgs) # length of the batch_sizeif training
        
    def __getitem__(self, idx):
        if self.train: 
            img = self.train_img
            msk = self.train_msk 
        else: 
            img_fname = self.train_imgs[idx] if self.train else self.val_imgs[idx]
            img_path = os.path.join(self.img_dir, img_fname)
            msk_path = os.path.join(self.msk_dir, img_fname)
            
            # load mask and image
            img = imread(img_path)
            msk = imread(msk_path)
            assert img.shape == msk.shape, '[error] expected img and msk sizes must match, img {}, msk {}'.format(img.shape, msk.shape)
            
            # normalize
            img = img / (2**16 - 1)
            msk = msk / (2**8 - 1)
            
            # add channel
            img = np.expand_dims(img, 0)
            msk = np.expand_dims(msk, 0)
            
            # to tensor
            img = torch.tensor(img, dtype=torch.float)
            msk = torch.tensor(msk, dtype=torch.float)

        # transform with TorchIO, tmp removed because using a resampled dataset
        transform = self.training_transform if self.train else self.validation_transform
        tmp_sub = transform(tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk)))
        img, msk = tmp_sub.img.tensor, tmp_sub.msk.tensor

        # images arcface
        img_fname = self.train_imgs[idx]
        img_path = os.path.join(self.img_dir, img_fname)
        
        # load mask and image
        img_arc = imread(img_path)
        
        # normalize
        img_arc = img_arc / (2**16 - 1)
        img_arc = np.expand_dims(img_arc, 0)
        img_arc = torch.tensor(img_arc, dtype=torch.float)

        # img = np.concatenate([img, img_arc])
        # img = torch.tensor(img, dtype=torch.float)
        # print(img.shape)
        # img_arc = torch.tensor(img_arc, dtype=torch.float)

        return img, msk, img_arc, idx


#---------------------------------------------------------------------------
# test 

def test_single():
    import matplotlib.pyplot as plt
    import napari
    t = Nucleus3DSingle(
        'data/nucleus/images_resampled_sophie',
        'data/nucleus/masks_resampled_sophie',
        'data/nucleus/folds_sophie.csv',
        train_idx= 0,
        batch_size= 4,
        train       =True,
        spacing     =(1,1,0.5),
        use_onehot  =False,
        use_aug= True,
        )

    imgs=t.__getitem__(3)
    viewer = napari.view_image(imgs[0].numpy(), name='img')
    viewer.add_labels(imgs[1].numpy().astype(int), name='msk')
    napari.run()

def test_semseg3d():
    import matplotlib.pyplot as plt
    import napari
    t = SemSeg3D(
        'data/nucleus/images_resampled_sophie',
        'data/nucleus/masks_resampled_sophie',
        'data/nucleus/folds_sophie.csv',
        fold     =0,
        train       =True,
        spacing     =(1,1,0.5),
        use_onehot  =False,
        )

    imgs=t.__getitem__(1)
    viewer = napari.view_image(imgs[0].numpy(), name='img')
    viewer.add_labels(imgs[1].numpy().astype(int), name='msk')
    napari.run()

def test_triplet():
    import napari

    t = Triplet(
        'data/nucleus/images_resampled_sophie',
        'data/nucleus/folds_sophie.csv',
        spacing     =(1,1,0.5),
        )
    imgs=t.__getitem__(1)
    viewer = napari.view_image(imgs[0].numpy(), name='anc')
    viewer.add_image(imgs[1].numpy(), name='pos')
    viewer.add_image(imgs[2].numpy(), name='neg')
    
    napari.run()

if __name__=='__main__':
    test_single()

#---------------------------------------------------------------------------