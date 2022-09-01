from random import shuffle
import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.ynet3d import *
from config import setup_config
from tqdm import tqdm
import tifffile
import scipy.ndimage
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from koila import LazyTensor, lazy
import numpy as np
import time
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

config = setup_config()
device = torch.device('cuda:0')
print(f"batch size = {config.batch_size}")

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

from datasets_old import SemSeg3D
from semseg_patch_fast import SemSeg3DPatchFast

train_dataset = SemSeg3DPatchFast('/mnt/sdb2/Adama/img',
                            '/mnt/sdb2/Adama/msk',
                            batch_size=config.batch_size, 
                            patch_size=(128, 128, 40),
                            nbof_steps=250,
                            folds_csv  = None, 
                            fold       = 0, 
                            val_split  = 0.2,
                            train      = True,
                            use_aug    = True,
                            fg_rate = 0.33, 
                            )

train_dataloader = DataLoader(train_dataset, batch_size=2, 
                              num_workers=4, 
                              pin_memory = True, 
                              shuffle=True)

# prepare the 3D model
model = UNet3D()

#Load pre-trained weights
weight_dir = 'Checkpoints/en_de/TransVW_chest_ct.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
delete = [key for key in state_dict if "projection_head" in key]
for key in delete: del state_dict[key]
delete = [key for key in state_dict if "prototypes" in key]
for key in delete: del state_dict[key]
for key in state_dict.keys():
    if key in model.state_dict().keys():
        model.state_dict()[key].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key, key))
    elif key.replace("classficationNet.", "") in model.state_dict().keys():
        model.state_dict()[key.replace("classficationNet.", "")].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key.replace("classficationNet.", ""), key))
    else:
        print("Key {} is not found".format(key))


model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = torch_dice_coef_loss
optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
print("\n")
print("====================")
print("Training in progress")
print("====================")
# Creates a GradScaler once at the beginning of training.
path_to_save_the_model = "/mnt/sdb2/Adama/configure_docker_for_transvw/pytorch/Task06_Lung"
scaler = GradScaler()
epoch = 0
train_start = time.time()
for epoch in tqdm(range(0, 1000)):
    epoch_start = time.time()
    model.train()
    print(f"Epoch {epoch}/1000 :")
    epoch = epoch + 1
    for (img, mask) in train_dataloader:
        optimizer.zero_grad()
        img = img.to(device)
        mask = mask.to(device)
        # Runs the forward pass with autocasting.
        with autocast():
            pred = model(img)
            loss = criterion(pred, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    epoch_end = time.time()
    print(f"Loss : {loss}")
    print(f"This epoch took {epoch_end - epoch_start}")
    if epoch%10 == 0 :
        print("Begin saving the model")
        torch.save(model, f"{path_to_save_the_model}/retrained_transvw_pretrained_model")
        print("End saving the model")
    print("\n")
train_end = time.time()
print(f"All the training took {train_end - train_start}")

#Save the model
print("\n")
print("=======================")
print("Saving the entire model")
print("=======================")
path_to_save_the_model = "/mnt/sdb2/Adama/configure_docker_for_transvw/pytorch/Task06_Lung"
torch.save(model, f"{path_to_save_the_model}/retrained_transvw_pretrained_model")

#model = torch.load(f"{path_to_save_the_model}/retrained_transvw_pretrained_model")
#model.eval()