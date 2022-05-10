
help() {
	echo "./run_prediction input_dir output_dir model_dir model_name"
	echo "input_dir: dataset input directory"
	echo "output_dir: dataset output directory"
	echo "model_dir: model directory"
	echo "model_name: model name"
	exit
}

if [ "$1" == "-h" -o "$1" == "--help" ]; then
	help
fi

input_dir="$1"
if [ -z "$1" ]; then
	input_dir="$(pwd)/dataset/in"
fi

output_dir="$2"
if [ -z "$2" ]; then
	output_dir="$(pwd)/dataset/out"
fi

model_dir="$3"
if [ -z "$3" ]; then
	model_dir="$(pwd)/dataset/model"
fi

model_name="$4"
if [ -z "$4" ]; then
	model_name="stardist"
fi

script="$(pwd)/basic_prediction.py"


docker run -it --rm -v "$input_dir":"$input_dir" -v "$output_dir":"$output_dir" -v "$model_dir":"$model_dir" -v "$script":"$script" stardist python $(pwd)/basic_prediction.py -i $input_dir -o $output_dir -m $model_dir -n $model_name
