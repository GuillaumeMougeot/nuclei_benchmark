#!/bin/sh
help() {
        echo "Format: ./run_evaluation_simple.sh true_absolute_path predicted_absolute_path name_true name_predicted"
        echo "true_absolute_path = /home/user/setup/dataset/true by default, expected results"
        echo "predicted_absoluite_path = /home/user/setup/dataset/predicted by default, predicted results"
        echo "name_true = true by default, true dataset name"
        echo "name_predicted = predicted by default, predicted dataset name"
        exit
}


if [ "$1" == "-h" -o "$1" == "--help" ]; then
        help
fi

in_dir="$1"
if [ -z "$1" ]; then
        in_dir="$(pwd)/dataset/true/"
fi

out_dir="$2"
if [ -z "$2" ]; then
        out_dir="$(pwd)/dataset/predicted/"
fi

name_true="$3"
if [ -z "$3" ]; then
	name_true="true"
fi

name_predicted="$4"
if [ -z "$4" ]; then
	name_predicted="predicted"
fi


docker run --rm -it -v "$(pwd)/":/work -v $in_dir:$in_dir -v $out_dir:$out_dir evaluatesegmentation ./EvaluateSegmentation.sh $in_dir $out_dir $name_true $name_predicted
