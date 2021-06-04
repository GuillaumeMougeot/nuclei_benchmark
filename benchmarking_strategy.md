# Benchmarking strategy



## git

### Add a new repository

1. Fork an interesting repo 
2. Add it in the git of the benchmarking as a submodule 
```
git submodule add <url> a/submodule 
```
3. Create a script to adapt the method on our dataset

### To clone this repo on a new machine

Option 1:
`git clone --recurse-submodules ...`

Option 2:
1. `git clone ...`
2. `git submodule update --init`

Option 3:
1. `git clone ...`
2. `git submodule init`
3. `git submodule update` to fetch changes from the submodule

### git pull for submodules

1. `git pull`
2. `git submodule update --init --recursive`

### push submodules

`git push --recurse-submodules=check`

## docker

1. [deprecated: This image is incompatible with rtx 3090 graphic cards] Pull the base tensorflow/pytorch environment with the jupyter notebook option 
```
docker run -it --name c_maskrcnn tensorflow/tensorflow:1.15.0-gpu-py3-jupyter 
```
1. Pull the base tensorflow image from the official nvidia website: https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
```
docker pull nvcr.io/nvidia/tensorflow:21.03-tf1-py3
```
2. Start the created container 
```
docker start c_maskrcnn 
```
3. Setup the container with the additional libraries in bash mode 
```
docker exec -it c_maskrcnn bash 
```
  * Then install the necessary python packages. NOTE: careful with the python installation, cf below 
  * For maskrcnn: 
    * [deprecated: do not use https://github.com/NVIDIA/tensorflow#install it does not work...] be sure to have python3.8 and pip20
    * if using NVidia Geforce RTX 3090 with 
    * ```pip install keras==2.0.8 numpy scipy Pillow matplotlib cython scikit-image==0.16.2 opencv-python h5py imgaug IPython[all]```
    * ```apt-get update```
    * ```apt-get install ffmpeg libsm6 libxext6 -y```
    * nvnidia
4. (optional) store the setup steps into a Dockerfile 
5. Commit the container to create a Docker image 
```
docker commit c_maskrcnn maskrcnn 
```
6. Run this image with the jupyter notebook and with the mounted local volume (cf. Command above) 
```
docker run -it --rm --runtime=nvidia -v $(realpath ~/all/codes/python):/tf/notebooks -p 8888:8888 maskrcnn 
```
Or with a (for a docker container not configure with jupyter notebook direct execution)
```
docker run -it --rm --runtime=nvidia \
       -v $(realpath ~/all/codes/python/3d_nuclei_seg_benchmark/2d_maskrcnn/Mask_RCNN):/tf/notebooks \
       -w /tf/notebooks \
       -p 8888:8888 \
       nv_maskrcnn jupyter notebook
```
6. (bis) Run this image with a python script only. The command below:
 * run docker image called 'maskrcnn'
 * in an interactive mode (you can see everything): '-it' flag
 * removes everything in the end: '--rm'
 * run the container with the GPU: '--runtime=nvidia'
 * mount the host directory to the container: '-v hostDir:containerDir'
 * set a working directory: '-w containerDir'
 * connect port: '-p hostPort:containerPort' (needed only if you want to use jupyter notebook)
 * run the docker image: 'nv_maskrcnn'
 * run the command into the container: 'python3 samples/nucleus/nucleus.py train --dataset=datasets/nucleus --subset=train --weights=imagenet'
```
docker run -it --rm --runtime=nvidia -v $(realpath ~/all/codes/python/3d_nuclei_seg_benchmark/2d_maskrcnn/Mask_RCNN):/tf/notebooks -w /tf/notebooks -p 8888:8888 nv_maskrcnn python3 samples/nucleus/nucleus.py train --dataset=datasets/nucleus --subset=train --weights=imagenet
```
7. (optional) upload the image on dockerhub: not sure, the image is really heavy: 15 Go!



