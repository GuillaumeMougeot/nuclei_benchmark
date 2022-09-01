#!/bin/bash

help() {
	echo "./generate_cubes.sh trainset_path valset_path"
	echo "trainset_path: path to dataset for training"
	echo "valset_path: path to dataset for validation"
    echo ""
	exit
}

if [ "$1" == "-h" -o "$1" == "--help" ]; then
    help
fi

if [[ "$#" -eq 0 || "$#" -eq 1 ]]
then
    echo "ERROR : Need command line arguments. --h or -help for help"
else

    # Checking if the trainset directory exists:
    if [[ ! -d $1 ]]; then
        echo "[-] Directory $cDir doesn't exist!"
        exit
    fi
    # Checking if the valset directory exists:
    if [[ ! -d $1 ]]; then
        echo "[-] Directory $cDir doesn't exist!"
        exit
    fi
    trainset="$1"
    validset="$2"

    # Generating npy file for training set
    echo "===================================="
    echo "Generating npy file for training set"
    echo "===================================="
    python -W ignore infinite_generator_3D.py \
    --fold 0 \
    --scale 32 \
    --data  $trainset\
    --save generated_cubes\
    --training train

    # Generating npy file for validation set
    echo "======================================"
    echo "Generating npy file for validation set"
    echo "======================================"
    python -W ignore infinite_generator_3D.py \
    --fold 0 \
    --scale 32 \
    --data  $validset\
    --save generated_cubes\
    --training valid
fi