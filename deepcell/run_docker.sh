docker run --gpus '"device=0"' -it --rm \
    -p 8888:8888 \
    -v $PWD/deepcell:/home/notebooks \
    -v $PWD/data:/home/data \
    -w /home \
    vanvalenlab/deepcell-tf:0.9.0-gpu
