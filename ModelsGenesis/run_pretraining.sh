#!/bin/bash

help() {
	echo "./run_pretraining.sh trainset_path valset_path"
	echo "trainset_path: path to dataset for training"
	echo "validset_path: path to dataset for validation"
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
    
    echo "======================="
    echo "Instal required modules"
    echo "======================="
    git clone https://github.com/MrGiovanni/ModelsGenesis.git
    pip3 install -r ModelsGenesis/requirements.txt
    pip3 install opencv-python
    pip3 install torchsummary

    # Generating cubes (npy files)
    ./generate_cubes.sh $trainset $validset

    # Pretraining Models Genesis
    echo "============================================"
    echo "The pretraining of Models Genesis is rinning"
    echo "============================================"
    
    rm ModelsGenesis/pytorch/Genesis_Chest_CT.py
    cp needed_files/Genesis_Chest_CT.py ModelsGenesis/pytorch/
    python -W ignore ModelsGenesis/pytorch/Genesis_Chest_CT.py
fi