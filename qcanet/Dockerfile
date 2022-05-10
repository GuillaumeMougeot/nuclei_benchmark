FROM python:3.6

RUN git clone -b pedro https://github.com/GuillaumeMougeot/QCANet.git

RUN curl -o /QCANet/models/learned_nsn.npz https://fun.bio.keio.ac.jp/software/QCANet/learned_nsn.npz

RUN curl -o /QCANet/models/learned_ndn.npz https://fun.bio.keio.ac.jp/software/QCANet/learned_ndn.npz

WORKDIR /QCANet

RUN pip install chainer matplotlib numpy scikit-image
