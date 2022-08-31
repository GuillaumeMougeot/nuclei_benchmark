#!/bin/bash

help() {
    echo ""
    echo "==========================="
    echo "Help for running ./build.sh"
    echo "==========================="
	echo "./build.sh imagesTr_path labelsTr_path output_path task_name"
	echo "imagesTr_path: path to train images input directory"
	echo "labelsTr_path: path to train labels input directory"
	echo "output_path  : path to the directory where you want to save predictions"
    echo "task_name    : the task name"
    echo "If you get an error telling you that an nnUNet project already exists, you need to delete the nnUNet project in your current directory"
    echo ""
	exit
}

if [ -e nnUNet ]
then
    echo "ERROR : There is already an nnUNet project. --h or -help for help"
else
    if [ "$1" == "-h" -o "$1" == "--help" ]; then
        help
    fi

    if [[ "$#" -eq 0 || "$#" -eq 1 || "$#" -eq 2 || "$#" -eq 3 ]]
    then
        echo "ERROR : Need command line arguments. --h or -help for help"
    else

        files=`ls $1`
        for item in $files; do
            if [ ${item: -3} = 'tif' ]
            then
                tif_or_nifti="tif"
            else 
                tif_or_nifti="nifti"
            fi
            break
        done

        # Checking if the training images directory exists:
        if [[ ! -d $1 ]]; then
            echo "[-] Directory $cDir doesn't exist!"
            exit
        fi
        # Checking if the training labels directory exists:
        if [[ ! -d $2 ]]; then
            echo "[-] Directory $cDir doesn't exist!"
            exit
        fi

        # Checking if the test images directory exists:
        if [[ ! -d $3 ]]; then
            echo "[-] Directory $cDir doesn't exist!"
            exit
        fi

        imagesTr_path="$1"
        labelsTr_path="$2"
        output_path="$3"

        git clone https://github.com/MIC-DKFZ/nnUNet.git

        cat nb_epochs.py >> nnUNet/nnunet/training/network_training/nnUNetTrainerV2.py
        cp utils.py nnUNet/nnunet
        cp Dockerfile nnUNet
        cp -r workdir nnUNet
        cd nnUNet
        docker build --tag nnunet .
        cd workdir
        ./configure_directories.sh "$4"

        if [ $tif_or_nifti = 'tif' ]
        then
            ./run.sh "$imagesTr_path" "$labelsTr_path" "$output_path"
        else
            ./run.sh "$imagesTr_path" "$labelsTr_path" "$imagesTr_path/../imagesTs" "$output_path"
        fi
    fi
fi