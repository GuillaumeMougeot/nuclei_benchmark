# command to run deep cell repo in a docker image
run this command from the nuclei_benchmark directory:
```
docker run --gpus '"device=0"' -it --rm \
    -p 8888:8888 \
    -v $PWD/deepcell/deepcell-tf/notebooks:/home/notebooks \
    -v $PWD/data:/home/data \
    -w /home \
    vanvalenlab/deepcell-tf:0.9.0-gpu
```
