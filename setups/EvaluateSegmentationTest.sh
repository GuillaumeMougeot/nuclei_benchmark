#!/bin/sh
# out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation mask_sophie.tif mask_guillaume.tif -use all)
sophie='/home/data/sophie/'
guillaume='/home/data/guillaume/'
otsu='/home/data/otsu/'
gift='/home/data/gift/'
mask_sophie='/home/data/masks_sophie'
mask_cellpose='/home/data/masks_resultat'
general_folder='/home/data/masks_sophie'


testTwoFolders() {
	out_val=''
	for f in $(ls $general_folder); do
        	out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation $1$f $2$f -use all)
        	out_val="$out_val$out \n"
	done
	echo $out_val >> $3_$4.txt

}

#####
#
#   testTwoFolders $path1 $path2 "name1" "name2"
#
#####


# guillaume vs sophie 
#testTwoFolders $guillaume $sophie "guillaume" "sophie"

# guillaume vs otsu
#testTwoFolders $guillaume $otsu "guillaume" "otsu"

# guillaume vs gift
#testTwoFolders $guillaume $gift "guillaume" "gift"



# sophie vs otsu
#testTwoFolders $sophie $otsu "sophie" "otsu"

# sophie vs gift
#testTwoFolders $sophie $gift "sophie" "gift"



# otsu vs gift
#testTwoFolders $otsu $gift "otsu" "gift"



# Mask sophie vs cellpose
testTwoFolders $mask_sophie $mask_cellpose "maskS" "cellpose"
