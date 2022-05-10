help() {
	echo "Format: ./run_predict in_absolute_path out_absolute_path predictpy_folder_abs_path"
	echo "in_absolute_path = /home/user/cellpose/dataset/in by default"
	echo "out_absoluite_path = /home/user/cellpose/dataset/out by default"
	echo "predictpy_folder_abs_path = /home/user/cellpose/ by default"
	exit
}


if [ "$1" == "-h" -o "$1" == "--help" ]; then
	help
fi

in_dir="$1"
if [ -z "$1" ]; then
	in_dir="$(pwd)/dataset/in"
fi

out_dir="$2"
if [ -z "$2" ]; then
        out_dir="$(pwd)/dataset/out"
fi
predict="$3"
if [ -z "$3" ]; then
	predict="$(pwd)"
fi


docker run --rm -v $in_dir:$in_dir -v $out_dir:$out_dir -v $predict:$predict cellpose:latest python $predict/predict.py -i $in_dir -o $out_dir

