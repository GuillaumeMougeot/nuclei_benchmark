# 3d_nuclei_seg_benchmark
Benchmarking of segmentation methods for 3D images of plant nuclei. Images are segmented. The segmentation is then evaluated with metrics (see 'metrics' section for more details).

## dataset
3D images of plant nuclei. An image contains a single nucleus. The ground truth segmentation was partially done thanks to NucleusJ and manually. 

## metrics
* 3D jaccard index.
* 3D dice index.
* Nucleusj morphology metrics. 

## setups
Each folder has to be setup with a Docker specific environment. 

## todo
### 2d (applied slice by slice)
1. [2d_maskrcnn] Mask-RCNN: https://github.com/matterport/Mask_RCNN
2. [2d_topcoders] DSB2018 winner team (3 codes available: selim, albu and victor): https://github.com/selimsef/dsb2018_topcoders 
3. [2d_stardist] Stardist: https://github.com/stardist/stardist  

### 3d
1. [3d_stardist] Stardist (no pretrained models...): https://github.com/stardist/stardist  

## work in progress
### 2d
1. [2d_maskrcnn] trying...
2. [2d_topcoders] DSB2018 winner team: selim's code outputs only black images (still work in progress)

## results
For now the results are computed on the OMERO_FSU dataset with the Otsu segmentation considered as the ground truth. This will be change later.

| Method           | 2D or 3D? | Framework | Avg. Jaccard   | Avg. Dice | Avg. F1   | NJ2      | Rmks           | 
|:----------------:|:---------:|:---------:|:--------------:|:---------:|:---------:|:--------:|:--------------:|
| topcoders_selim  | 2D        | tf1_keras | --             | --        | --        | --       | Empty outputs  |
| maskrcnn         | 2D        | tf1_keras | 0.380          | 0.504     | --        | --       | the model was pre-trained on DSB2018 with custom parameters  |
| deepcell         | 3D        | tf2       | 0.327          | 0.446     | --        | --       | selection of only the biggest labeled object |
