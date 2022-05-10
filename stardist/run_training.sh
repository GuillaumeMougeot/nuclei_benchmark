help() {
	echo "./run_prediction csv_file_path img_dir mask_dir model_out_dir model_name"
	echo "csv_file_path: path to a csv file that classifies the dataset between train and validation"
	echo "img_dir: dataset image directory"
	echo "mask_dir: dataset mask directory"
	echo "model_out_dir: model output directory"
	echo "model_name: model name"
	exit
}

if [ "$1" == "-h" -o "$1" == "--help" ]; then
	help
fi

csv_file_path="$1"
if [ -z "$1" ]; then
	csv_file_path="$(pwd)/dataset/folds_x_sophie.csv"
fi


img_dir="$2"
if [ -z "$2" ]; then
	img_dir="$(pwd)/dataset/in"
fi

mask_dir="$3"
if [ -z "$3" ]; then
	mask_dir="$(pwd)/dataset/masks"
fi

model_out_dir="$4"
if [ -z "$4" ]; then
	model_out_dir="$(pwd)/dataset/model"
fi

model_name="$5"
if [ -z "$5" ]; then
	model_name="stardist_model"
fi

script="$(pwd)/basic_training.py"


docker run -it --rm -v "$csv_file_path":"$csv_file_path" -v "$img_dir":"$img_dir" -v "$mask_dir":"$mask_dir" -v "$model_out_dir":"$model_out_dir" -v "$script":"$script" stardist python $(pwd)/basic_training.py -c $csv_file_path -i $img_dir -m $mask_dir -o $model_out_dir -n $model_name

