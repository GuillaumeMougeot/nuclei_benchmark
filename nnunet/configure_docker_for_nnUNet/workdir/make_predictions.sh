#!/bin/bash

help() {
	echo "./make_predictions.sh output_path task_name"
	echo "model_path: path to the directory which contain the trained model"
	echo "imagesTs_path: path to the directory of images oyu want to use for prediction"
	echo "output_path: path to the directory where you want to save predictions"
    echo "task_name  : the task name (default value = Nucleus)"
    echo ""
    echo "If you get an error telling you that an nnUNet project already exists, you need to delete the nnUNet project in your current directory"
	exit
}

if [ "$1" == "-h" -o "$1" == "--help" ]; then
    help
fi

if [[ "$#" -eq 0 || "$#" -eq 1 || "$#" -eq 2 || "$#" -eq 2 ]]
then
    echo "ERROR : Need command line arguments. --h or -help for help"
else :
    # Checking if the directory exists:
    if [[ ! -d $1 ]]; then
        echo "[-] Directory $cDir doesn't exist!"
        exit
    fi

    # Checking if the directory exists:
    if [[ ! -d $2 ]]; then
        echo "[-] Directory $cDir doesn't exist!"
        exit
    fi

    # Checking if the directory exists:
    if [[ ! -d $3 ]]; then
        echo "[-] Directory $cDir doesn't exist!"
        exit
    fi

    task_name="$4"

    rm -r nnUNet_trained_models
    cp -r "$1/nnUNet_trained_models" .

    rm nnUNet_raw_data_base/nnUNet_raw_data/Task500_"$task_name"/imagesTs/*
    cp "$2/*" nnUNet_raw_data_base/nnUNet_raw_data/Task500_"$task_name"/imagesTs

    export nnUNet_raw_data_base="nnUNet_raw_data_base"
    export nnUNet_preprocessed="nnUNet_preprocessed"
    export RESULTS_FOLDER="nnUNet_trained_models"

    echo "======================================="
    echo "Running inference on your test images"
    echo "======================================="

    echo ""
    echo "============================="
    echo "Running inference for fold 0"
    echo "============================="
    nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task500_"$task_name"/imagesTs/ -o output_for_fold_0 -t 500 -tr nnUNetTrainer_Experimental -m 3d_fullres -f 0

    echo ""
    echo "============================="
    echo "Running inference for fold 1"
    echo "============================="
    nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task500_"$task_name"/imagesTs/ -o output_for_fold_1 -t 500 -tr nnUNetTrainer_Experimental -m 3d_fullres -f 1

    echo ""
    echo "============================="
    echo "Running inference for fold 2"
    echo "============================="
    nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task500_"$task_name"/imagesTs/ -o output_for_fold_2 -t 500 -tr nnUNetTrainer_Experimental -m 3d_fullres -f 2

    echo ""
    echo "============================="
    echo "Running inference for fold 3"
    echo "============================="
    nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task500_"$task_name"/imagesTs/ -o output_for_fold_3 -t 500 -tr nnUNetTrainer_Experimental -m 3d_fullres -f 3

    echo ""
    echo "============================="
    echo "Running inference for fold 4"
    echo "============================="
    nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task500_"$task_name"/imagesTs/ -o output_for_fold_4 -t 500 -tr nnUNetTrainer_Experimental -m 3d_fullres -f 4

    echo ""
    echo "======================="
    echo "Running final inference"
    echo "======================="
    nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task500_"$task_name"/imagesTs/ -o output_for_all_folds -t 500 -tr nnUNetTrainer_Experimental -m 3d_fullres

    echo ""
    echo "======================="
    echo "Converting your data from nifti to tif"
    echo "======================="
    pip install torchio
    python3 post_processing.py -i output_for_all_folds -o "$2"
fi