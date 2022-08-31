#!/bin/bash

help() {
	echo "./nifti_to_tiff input output"
	echo "input : path to nifti images to convert"
    echo "output: path to save tiff images"
    echo ""
	exit
}

if [ "$1" == "-h" -o "$1" == "--help" ]; then
    help
fi

if [[ "$#" -eq 0 || "$#" -eq 1 ]]
then
    echo "ERROR : Need command line arguments. --h or -help for help"
else :
    if [ -e process_nifti_to_tiff.py ]
    then
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
        echo "===================================="
        echo "Converting images from nifti to tiff"
        echo "===================================="    
        #pip install torchio
        python3 process_nifti_to_tiff.py -i "$1" -o "$2"
    else
        echo "Le fichier process_tiff_to_nifti.py doit etre dans le meme répertoire que l'exécutable"
    fi
fi