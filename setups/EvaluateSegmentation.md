# Evaluate segmentation

I found a repository to evaluate the different segmentation metrics over two segmentations results
[link](https://github.com/Visceral-Project/EvaluateSegmentation).

## install

I installed the docker image with (where 'EvaluateSegmentation' is the git folder containing the Dockerfile):
```
docker build EvaluateSegmentation
```

A small bug in the installation was that the 'EvaluateSegmentation' command was not properly linked. This command is stored in the docker container in:
```
/opt/evaluate-segmentation/build/EvaluateSegmentation
```

## usage

### To run the docker container (generic):
```
docker run --gpus all -it --rm -v /path/to/image/folder/:/home/data -w /home --ipc=host seg-metrics
```

or (on my computer):
```
docker run --gpus all -it --rm -v /home/gumougeot/all/data/3d_nucleus/all_manual/single/:/home/data -w /home --ipc=host seg-metrics
```

### Then run the command in the docker container:
```
/opt/evaluate-segmentation/build/EvaluateSegmentation ground_truth_mask.tif prediction_mask.tif -use all
```

or stores this command in a '.sh' file and run it. It will output all the possible metric for this image in the shell.

If using a .sh file then be careful to apply 'chmod +x cmd.sh' to allow its execution.

### Additional remarks
To store the shell results using a .sh file just do:
```
a = $(./cmd.sh)
```
