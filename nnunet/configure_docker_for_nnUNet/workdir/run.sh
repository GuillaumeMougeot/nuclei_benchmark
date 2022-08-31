#!/bin/bash
if [[ "$#" -eq 3 ]]
then
    docker run -it --gpus all -v "$1":"$1" -v "$2":"$2" -v "$3":"$3" -v "$(pwd)":"$(pwd)" -w "$(pwd)" -p 8896:8888 --ipc=host nnunet
else :
    docker run -it --gpus all -v "$3/..":"$3/.." -v "$(pwd)":"$(pwd)" -w "$(pwd)" -p 8896:8888 --ipc=host nnunet
fi
