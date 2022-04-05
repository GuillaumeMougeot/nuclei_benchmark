# out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation mask_sophie.tif mask_guillaume.tif -use all)
sophie='/home/data/sophie/'
guillaume='/home/data/guillaume/'
otsu='/home/data/otsu/'
gift='/home/data/gift/'

# guillaume vs sophie
out_val=''
for f in $(ls /home/data/sophie)
do
	out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation $guillaume$f $sophie$f -use all)
	out_val="$out_val$out \n "
done
echo $out_val >> guillaume_sophie.txt

# guillaume vs otsu
out_val=''
for f in $(ls /home/data/sophie)
do
        out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation $guillaume$f $otsu$f -use all)
        out_val="$out_val$out \n "
done
echo $out_val >> guillaume_otsu.txt

# guillaume vs gift
out_val=''
for f in $(ls /home/data/sophie)
do
        out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation $guillaume$f $gift$f -use all)
        out_val="$out_val$out \n "
done
echo $out_val >> guillaume_gift.txt

# sophie vs otsu
out_val=''
for f in $(ls /home/data/sophie)
do
        out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation $sophie$f $otsu$f -use all)
        out_val="$out_val$out \n "
done
echo $out_val >> sophie_otsu.txt

# sophie vs gift
out_val=''
for f in $(ls /home/data/sophie)
do
        out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation $sophie$f $gift$f -use all)
        out_val="$out_val$out \n "
done
echo $out_val >> sophie_gift.txt

# otsu vs gift
out_val=''
for f in $(ls /home/data/sophie)
do
        out=$(/opt/evaluate-segmentation/build/EvaluateSegmentation $otsu$f $gift$f -use all)
        out_val="$out_val$out \n "
done
echo $out_val >> otsu_gift.txt


