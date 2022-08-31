FROM ubuntu

RUN apt-get update

RUN apt-get install -y \
        python3 \
        python3-pip \
        cmake \
        g++ \
        gcc \
        git \
        make \
        perl \
        wget \
        && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/usr/lib/python3.8/site-packages

# Copy the rest of the package code and its scripts
COPY . /opt/nnUNet

#Install Pytorchdocker images
RUN pip3 install torch torchvision torchaudio

# Install nnUNet
RUN pip install git+https://github.com/MIC-DKFZ/batchgenerators.git
RUN pip install -e /opt/nnUNet/

#RUN export nnUNet_raw_data_base="nnUNet_raw_data_base"
#RUN export nnUNet_preprocessed="nnUNet_preprocessed"
#RUN export RESULTS_FOLDER="nnUNet_trained_models"

#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]

#nnUNet_plan_and_preprocess -t xxx --verify_dataset_integrity




#A modifier. On ne veut pas de tensorflow
#FROM python:3.9

#RUN apt-get install python3.9

#WORKDIR /notebooks

#RUN git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/nnUNet/

# Copy the rest of the package code and its scripts
#COPY . /opt/nnUNet

# Install nnUNet
#RUN pip install -e /opt/nnUNet/

#Il manque le fichier utils.py quand je clone le dépot
#COPY utils.py /opt/nnUNet/nnunet

#Install Pytorch
#RUN pip3 install torch torchvision torchaudio

#Install torchio
#RUN pip install torchio

#Setting environnement variables
#RUN export nnUNet_raw_data_base="nnUNet_raw_data_base"
#RUN export nnUNet_preprocessed="nnUNet_preprocessed"
#RUN export RESULTS_FOLDER="nnUNet_trained_models"

#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]