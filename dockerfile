FROM ubuntu:bionic

WORKDIR /

# install pre-requisite packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            cmake \
            lsb-release \
            gnupg2 \
            python3-pip

# pip dependencies
RUN pip3 install --upgrade pip &&\
    pip3 install --verbose --upgrade Cython &&\
    pip3 install numpy \
                opencv-python \
                tqdm \
                scipy \