#!/bin/bash

help() {
    echo ""
    echo "================================"
    echo "Help for running train_nnUNet.sh"
    echo "================================"
	echo "./train_nnUNet.sh imagesTr_path labelsTr_path output_path task_name"
	echo "imagesTr_path: path to train images input directory"
	echo "labelsTr_path: path to train labels input directory"
	echo "output_path  : path to the directory where you want to save predictions"
    echo "task_name    : the task name"
    echo "If you get an error telling you that an nnUNet project already exists, you need to delete the nnUNet project in your current directory"
    echo ""
	exit
}

if [ "$1" == "-h" -o "$1" == "--help" ]; then
    help
fi

if [[ "$#" -eq 0 || "$#" -eq 1 || "$#" -eq 2 || "$#" -eq 3 ]]
then
    echo "ERROR : Need more command line arguments. --h or -help for help"
else :

    export nnUNet_raw_data_base="nnUNet_raw_data_base"
    export nnUNet_preprocessed="nnUNet_preprocessed"
    export RESULTS_FOLDER="nnUNet_trained_models"
    
    # Checking if imagsTr the directory exists:
    if [[ ! -d $1 ]]; then
        echo "[-] Directory $cDir doesn't exist!"
        exit
    fi

    # Checking if the labelsTr directory exists:
    if [[ ! -d $2 ]]; then
        echo "[-] Directory $cDir doesn't exist!"
        exit
    fi

    # Checking if the labelsTr directory exists:
    if [[ ! -d $3 ]]; then
        echo "[-] Directory $cDir doesn't exist!"
        exit
    fi

    task_name="$4"

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

    if [ $tif_or_nifti = 'tif' ]
    then
        echo "Les images sont au format tiff"
        echo ""
        echo "======================================="
        echo "Converting your data from tiff to nifti"
        echo "======================================="
        python3 pre_processing.py -i "$1" -l "$2" -t "$task_name"
    else
        echo "Les images sont au format nifti"
        #python3 train_test_split.py -i "$1" -l "$2" -t "$task_name"
        cp -r $1/* nnUNet_raw_data_base/nnUNet_raw_data/Task500_$task_name/imagesTr
        cp -r $2/* nnUNet_raw_data_base/nnUNet_raw_data/Task500_$task_name/labelsTr
        cp -r $2/../imagesTs/* nnUNet_raw_data_base/nnUNet_raw_data/Task500_$task_name/imagesTs
        cp $2/../dataset.json nnUNet_raw_data_base/nnUNet_raw_data/Task500_$task_name
    fi

    echo ""
    echo "============================="
    echo "Running nnnUNet preprocessing"
    echo "============================="
    nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity

    #deleteting all .npy files in the nnUNet_preprocessed
    rm -f nnUNet_preprocessed/Task500_"$task_name"/nnUNetData_plans_v2.1_stage0/*.npy

    echo ""
    echo "============================"
    echo "Training nnUNet on your data"
    echo "============================"
    for FOLD in 0 1 2 3 4
    do
        nnUNet_train 3d_fullres nnUNetTrainer_Experimental Task500_"$task_name" "$FOLD" --npz --use_compressed
    done

    #Saving the trained model in the output_directory
    cp -r nnUNet_trained_models "$3"
fi