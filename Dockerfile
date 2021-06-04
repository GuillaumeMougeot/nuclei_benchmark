FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter

WORKDIR /code

RUN pip install keras==2.0.8 numpy scipy Pillow matplotlib cython scikit-image==0.16.2 opencv-python h5py imgaug IPython[all]

# RUN apt-get install git wget 
# RUN apt-get update
# RUN apt-get install git wget 

COPY . /code


# CMD ["python3"]

