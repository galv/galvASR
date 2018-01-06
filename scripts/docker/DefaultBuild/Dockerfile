FROM nvidia/cuda:9.1-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        python3-dev \
        clang \
        cmake \
        git \
        gfortran-multilib \
        libatlas-base-dev \
        liblapacke-dev \
        pkg-config \
        wget \
        curl \
        zlib1g-dev \
        ca-certificates \
        # For Kaldi
        python-dev \
        automake \
        libtool \
        autoconf \
        subversion

# Install bazel
RUN apt-get install -y --no-install-recommends openjdk-8-jdk && \
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" > /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends bazel

# We can't use the Ubuntu cudnn docker container because tensorflow is
# dumb and can't figure out CUDNN's install path that way (as far as I
# know).
RUN CUDNN_DOWNLOAD_SUM=1ead5da7324db35dcdb3721a8d4fc020b217c68cdb3b3daa1be81eb2456bd5e5 && \
    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-9.1-linux-x64-v7.tgz -O && \
    echo "$CUDNN_DOWNLOAD_SUM  cudnn-9.1-linux-x64-v7.tgz" | sha256sum -c - && \
    tar --no-same-owner -xzf cudnn-9.1-linux-x64-v7.tgz -C /usr/local && \
    rm cudnn-9.1-linux-x64-v7.tgz && \
    ldconfig

