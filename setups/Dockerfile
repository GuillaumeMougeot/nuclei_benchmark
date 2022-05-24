# Dockerfile for EvaluateSegmentation.
#
# Build image:
#   docker build --tag evaluatesegmentation .
#
# Run image:
#   docker run --rm -it -v $PWD:/data evaluatesegmentation /data/reference.nii.gz /data/segmentation.nii.gz

ARG MAKE_JOBS=1

FROM ubuntu:20.04 AS base
FROM base AS builder

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y \
        python3.8 \
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
RUN pip3 install pandas numpy

ARG MAKE_JOBS
WORKDIR /opt/ITK
RUN wget -O- -q https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.1.1/InsightToolkit-5.1.1.tar.gz \
    | tar xz --strip 1 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make --jobs "$MAKE_JOBS" \
    && make install

WORKDIR /opt/evaluate-segmentation
RUN git clone https://github.com/Visceral-Project/EvaluateSegmentation.git . \
    && mkdir build \
    && cd build \
    && cmake -DITK_DIR=/usr/local/lib/cmake/ITK-5.1 ../source \
    && make --jobs "$MAKE_JOBS"

FROM base
COPY --from=builder /opt/evaluate-segmentation/build/EvaluateSegmentation /usr/local/bin/
# RUN apk add --no-cache libgcc libstdc++
WORKDIR /work
RUN apt-get update
RUN apt-get install -y \
	python3.8 \
	python3-pip
RUN pip3 install pandas numpy
#ENTRYPOINT ["/usr/local/bin/EvaluateSegmentation"]
#CMD ["--help"]
