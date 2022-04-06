#!/bin/sh
# out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation mask_sophie.tif mask_guillaume.tif -use all)

sophie='/home/data/sophie/'
guillaume='/home/data/guillaume/'
otsu='/home/data/otsu/'
gift='/home/data/gift/'
#mask_sophie='/home/data/masks_sophie/'
cellpose='/home/data/masks_resultat/'

general_folder='/home/data/sophie'

all_img=""
all_img="${all_img} ${sophie}"
all_img="${all_img} ${guillaume}" 
all_img="${all_img} ${otsu}"
all_img="${all_img} ${gift}" 
#all_img="${all_img} ${mask_sophie}"
all_img="${all_img} ${cellpose}"


all_tags=""
all_tags="${all_tags} sophie" 
all_tags="${all_tags} guillaume" 
all_tags="${all_tags} otsu" 
all_tags="${all_tags} gift" 
#all_tags="${all_tags} maskS" 
all_tags="${all_tags} cellpose"

total=$(($(echo $all_tags | wc -w)+1))

testTwoFolders() {
	out_val=''
	for f in $(ls $general_folder); do
        	out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation $1$f $2$f -use all)
        	out_val="$out_val$out \n"
	done
	echo $out_val >> results/$3_$4.txt

}


i=1
for path1 in $all_img; do
	j=$(($i+1))
	while [ $j -lt ${total} ]; do
		path2=$(echo $all_img | cut -d' ' -f$j)
		name1=$(echo $all_tags | cut -d' ' -f$i)
		name2=$(echo $all_tags | cut -d' ' -f$j)
		
		if [ ! -f "results/${name1}_${name2}.txt" ] && [ ! -f "results/${name2}_${name1}.txt" ]; then
			echo "comparison between ${name1} and ${name2}"
			testTwoFolders ${path1} ${path2} ${name1} ${name2}
		else
			echo "results/${name1}_${name2}.txt already exists"
		fi	
		echo
		j=$(($j+1))
	done
	i=$(($i+1))
done





#####
#
#   testTwoFolders $path1 $path2 "name1" "name2"
#
#####





# guillaume vs sophie 
#testTwoFolders $guillaume $sophie "guillaume" "sophie"
#testTwoFolders $sophie $guillaume "sohpie" "guillaume"


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
#testTwoFolders $mask_sophie $mask_cellpose "maskS" "cellpose"
