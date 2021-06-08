# command to run deep cell repo in a docker image
run this command from the 3d_nuclei_seg_benchmark directory:
```
docker run --gpus '"device=0"' -it --rm \
    -p 8888:8888 \
    -v $PWD/notebooks:/home/notebooks \
    -v $PWD/data:/home/data \
    -w /home \
    vanvalenlab/deepcell-tf:0.9.0-gpu
```
