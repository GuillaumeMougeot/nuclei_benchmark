# MaskRCNN setup

### hardware requirements

* one recent Nvidia GPU
* docker installed

### pull nvidia-tf1 docker image and run it as a container 

 docker run -it -d --name maskrcnn nvcr.io/nvidia/tensorflow:21.11-tf1-py3

### open docker container in bash mode ...

 docker exec -it maskrcnn bash 

### ... and install the pip requirements in it

 pip install keras==2.0.8 numpy scipy Pillow matplotlib cython scikit-image==0.16.2 opencv-python h5py imgaug IPython[all]
 apt-get update
 apt-get install ffmpeg libsm6 libxext6 -y

### quit the docker container by pressing CTRL+D

### commit it as a novel docker image

 docker commit maskrcnn maskrcnn

### remove the container

 docker rm maskrcnn

### run the novel image from the nuclei_benchmark directory

 docker run --name maskrcnn -v $PWD:/home -w /home maskrcnn

### go to maskrcnn folder and run the python installer of the maskrcnn repository

 cd maskrcnn/Mask_RCNN/

 pip install .

### quit the docker container by pressing CTRL+D

### commit it again

 docker commit maskrcnn maskrcnn

### installation done. the docker can now be used with the following command from the nuclei_benchmark repo

 docker run --gpus all -it --rm -v $PWD:/home -w /home -p 8888:8888 maskrcnn 

or to execute a jupyter notebook directly:

 docker run --gpus all -it --rm -v $PWD:/home -w /home -p 8888:8888 maskrcnn jupyter notebook

### run predictions with maskrcnn on a custom dataset of 3D nuclei

predictions can now be run from the notebook located at ```maskrcnn/MaskRCNN/samples/nucleus/3d_nucleus_run_prediction.ipynb```

The model must be pretrained on a nucleus dataset (it can be the base DSB2018 dataset)

 

