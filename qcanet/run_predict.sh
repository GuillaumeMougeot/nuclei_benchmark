help() {
        echo "Format: ./run_predict in_path out_path"
        echo "in_path = /home/user/qcanet/dataset/in by default"
        echo "out_path = /home/user/cellpose/dataset/out by default"
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

if [ ! -e "$(pwd)/qcanet/models/learned_nsn.npz" ]; then
	curl -o "$(pwd)/qcanet/models/learned_nsn.npz" https://fun.bio.keio.ac.jp/software/QCANet/learned_nsn.npz
fi

if [ ! -e "$(pwd)/qcanet/models/learned_ndn.npz" ]; then
	curl -o "$(pwd)/qcanet/models/learned_ndn.npz" https://fun.bio.keio.ac.jp/software/QCANet/learned_ndn.npz
fi

#rm -rf "$(pwd)"/qcanet/results/*
#cp "$in_dir"/* "$(pwd)"/qcanet/images/example_input/

docker run -it --rm -v "$in_dir/":/QCANet/images/example_input/ -v "$out_dir/":/QCANet/results -v "$(pwd)/basic_predict.py:/QCANet/basic_predict.py" qcanet:latest # python /QCANet/basic_predict.py

#cp -rf "$(pwd)"/qcanet/results/* "$out_dir/"
#rm "$(pwd)"/qcanet/images/example_input/*
