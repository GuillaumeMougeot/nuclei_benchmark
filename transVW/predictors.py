#---------------------------------------------------------------------------
# model predictors
# these functions returns the model predictions for a single input 
# TODO: re-structure with classes maybe?
#---------------------------------------------------------------------------

import torch 
import torchio as tio
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

import utils

#---------------------------------------------------------------------------
# model predictor for segmentation

def load_img_seg(fname):
    img = imread(fname)
    # img = utils.sitk_imread()(fname)

    # normalize
    img = (img - img.min()) / (img.max() - img.min())
    # img = (img-img.mean())/img.std()
    
    # to tensor
    img = torch.tensor(img, dtype=torch.float)

    # expand dim
    img = torch.unsqueeze(img, dim=0)
    img = torch.unsqueeze(img, dim=0)

    return img

def seg_predict(
    img_path,
    model,
    return_logit=False,
    ):
    """
    for one image path, load the image, compute the model prediction, return the prediction
    """
    img = load_img_seg(img_path)
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        logit = model(img)[0][0]

    if return_logit: return logit.cpu().detach().numpy()

    out = (torch.sigmoid(logit)>0.5).int()*255
    return out.cpu().detach().numpy()

def seg_predict_old(img, model, return_logit=False):
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        img = torch.unsqueeze(img, 0)
        logit = model(img)[0]

    if return_logit: return logit.cpu().detach().numpy()

    out = (torch.sigmoid(logit)>0.5).int()*255
    return out.cpu().detach().numpy()

#---------------------------------------------------------------------------
# model predictor for segmentation with patches

def load_img_seg_patch(fname, patch_size=(64,64,32)):
    """
    Prepare image for model prediction
    """
    # load the image
    # img = imread(fname)
    img = utils.sitk_imread()(fname)

    # normalize the image
    # bits = lambda x: ((np.log2(x)>8).astype(int)+(np.log2(x)>16).astype(int)*2+1)*8
    # img = img / (2**bits(np.max(img)) - 1)
    img = (img-img.mean())/img.std()
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # define the grid sampler 
    sub = tio.Subject(img=tio.ScalarImage(tensor=img))
    patch_size = np.array(patch_size)
    sampler= tio.data.GridSampler(subject=sub, 
                            patch_size=patch_size, 
                            patch_overlap=patch_size//2,
                            padding_mode='constant')
    return sampler

def seg_predict_patch(
    img_path,
    model,
    return_logit=False,
    patch_size=None,
    tta=False,          # test time augmentation 
    ):
    """
    for one image path, load the image, compute the model prediction, return the prediction
    """
    img = load_img_seg_patch(img_path, patch_size)
    model.eval()
    with torch.no_grad():
        pred_aggr = tio.inference.GridAggregator(img, overlap_mode='hann')
        patch_loader = torch.utils.data.DataLoader(
            img, 
            batch_size=4, 
            drop_last=False, 
            shuffle  =False, 
            num_workers=4, 
            pin_memory =True)

        for patch in tqdm(patch_loader):
            X = patch['img'][tio.DATA]
            if torch.cuda.is_available():
                X = X.cuda()
            
            if tta: # test time augmentation: flip around each axis
                Xs = [X] + [torch.flip(X,dims=[i]) for i in range(2,5)]
                with torch.cuda.amp.autocast():
                    preds = [model(x).detach() for x in Xs]
                preds = [preds[0]]+[torch.flip(preds[i], dims=[i+1]) for i in range(1,4)]
                pred = torch.mean(torch.stack(preds), dim=0)
            else:
                with torch.cuda.amp.autocast():
                    pred=model(X).detach()
            pred_aggr.add_batch(pred, patch[tio.LOCATION])
        
        logit = pred_aggr.get_output_tensor().float()
    
    torch.cuda.empty_cache()

    if return_logit: return logit.cpu().detach().numpy()

    # out = (logit.softmax(dim=0).argmax(dim=0)).int()
    out = (torch.sigmoid(logit)>0.5).int()
    out = out.cpu().detach().numpy()
    out = out.astype(np.byte)
    # out = utils.keep_center_only(out)
    print("output shape",out.shape)
    return out

#---------------------------------------------------------------------------
