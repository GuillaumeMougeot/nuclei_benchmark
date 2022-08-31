#!/bin/bash

mkdir -p nnUNet_preprocessed
mkdir -p nnUNet_trained_models
mkdir -p nnUNet_raw_data_base/nnUNet_raw_data
mkdir -p nnUNet_raw_data_base/nnUNet_raw_data/"Task500_$1"
mkdir -p nnUNet_raw_data_base/nnUNet_raw_data/"Task500_$1"/imagesTr
mkdir -p nnUNet_raw_data_base/nnUNet_raw_data/"Task500_$1"/imagesTs
mkdir -p nnUNet_raw_data_base/nnUNet_raw_data/"Task500_$1"/labelsTr
mkdir -p nnUNet_raw_data_base/nnUNet_raw_data/"Task500_$1"/labelsTs
mkdir -p output_for_all_folds
mkdir -p output_for_fold_0
mkdir -p output_for_fold_1
mkdir -p output_for_fold_2
mkdir -p output_for_fold_3
mkdir -p output_for_fold_4