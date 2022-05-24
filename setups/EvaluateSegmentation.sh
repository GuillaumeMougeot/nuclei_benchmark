#!/bin/sh
apk add --update --no-cache py3-numpy py3-pandas@testing

out_val=''
for f in $(ls $1)
do
	out=$(EvaluateSegmentation "$1/$f" "$2$f" -use all)
	out_val="$out_val$out \n "
done
echo $out_val >> $3_$4.txt
mv $3_$4.txt dataset/metrics/
python3 read_evaluate-segmentation_output.py -i $3_$4 -d "/work/dataset/metrics/"
