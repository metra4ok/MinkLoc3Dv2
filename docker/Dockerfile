FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    sudo \
    openssh-server \
    ninja-build \
    cmake \
    libopenblas-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools

# Install MinkowskiEngine
ENV MAX_JOBS=4
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"

RUN conda update -y conda
RUN conda install -y ipykernel

# install conda requirements
RUN conda install -y \
        pytorch-metric-learning \
        faiss-gpu \
        tensorboard \
        wandb \
        pandas \
        -c metric-learning -c pytorch -c conda-forge

# some requirements are not available in conda
RUN pip install colour-demosaicing

# add user and his password
ARG USER=docker_minkloc3d
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}

RUN mkdir MinkLoc3Dv2 && chown -R ${UID}:${GID} /home/${USER}
RUN mkdir Datasets && chown -R ${UID}:${GID} /home/${USER}

USER ${UID}:${GID}