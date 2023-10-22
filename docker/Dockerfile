ARG NCORE

ARG BLD_PINOCCHIO_URL=https://github.com/stack-of-tasks/pinocchio
ARG BLD_PINOCCHIO_VERSION=e7f00457b5928e2f9e0a4567da2c82b5abbdc93f
ARG BLD_CMAKE_VERSION=3.27.0
ARG BLD_CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${BLD_CMAKE_VERSION}/cmake-${BLD_CMAKE_VERSION}.tar.gz
ARG BLD_EIGEN_URL=https://gitlab.com/libeigen/eigen.git
ARG BLD_EIGEN_VERSION=3.4.0
ARG BLD_URDFDOM_HEADERS_URL=https://github.com/ros/urdfdom_headers.git
ARG BLD_URDFDOM_HEADERS_VERSION=00c1c9c231e46b2300d04073ad696521758fa45c
ARG BLD_URDFDOM_URL=https://github.com/ros/urdfdom.git
ARG BLD_URDFDOM_VERSION=0da4b20675cdbe14b532d484a1c17df85b4e1584
ARG BLD_TINYXML_URL=https://git.code.sf.net/p/tinyxml/git
ARG BLD_BOOST_URL=https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.bz2
ARG BLD_EIGENPY_URL=https://github.com/stack-of-tasks/eigenpy.git

# FROM ubuntu:22.04
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

ARG NCORE
ARG BLD_PINOCCHIO_URL
ARG BLD_PINOCCHIO_VERSION
ARG BLD_CMAKE_URL
ARG BLD_CMAKE_VERSION
ARG BLD_EIGEN_URL
ARG BLD_EIGEN_VERSION
ARG BLD_EIGENPY_URL
ARG BLD_URDFDOM_HEADERS_URL
ARG BLD_URDFDOM_HEADERS_VERSION
ARG BLD_URDFDOM_URL
ARG BLD_URDFDOM_VERSION
ARG BLD_TINYXML_URL
ARG BLD_BOOST_URL
ARG BLD_MINICONDA_URL

ARG DEBIAN_FRONTEND=noninteractive

ENV INSTALLDIR=/root

RUN cat /etc/resolv.conf

RUN apt-get update \
    && apt-get install \
    sudo \
    git \
    git-lfs \
    ssh \
    wget \
    build-essential \
    libssl-dev \
    libglfw3 \
    apt-transport-https ca-certificates \
        --assume-yes --no-install-recommends && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CC="gcc"
ENV CXX="g++"


# CMake
RUN cd /root \
    && wget --quiet ${BLD_CMAKE_URL} \
    && tar -xf cmake-${BLD_CMAKE_VERSION}.tar.gz \
    && cd /root/cmake-${BLD_CMAKE_VERSION} \
    && ./bootstrap \
    && make -j ${NCORE} \
    && make install \
    && cd /root && rm -rf /root/cmake-${BLD_CMAKE_VERSION}

# Install boost
RUN cd /root && wget -q -c ${BLD_BOOST_URL} -O /root/boost.tar.bz2 \
        && mkdir boost && cd /root/boost && tar -xf /root/boost.tar.bz2 --strip-components=1 && \
     cd /root/boost && \
          ./bootstrap.sh && \
           ./b2 -j ${NCORE} install ; cd /root; rm -rf /root/boost*

# Eigen3
RUN cd $RH_INSTALLDIR \
        && git clone ${BLD_EIGEN_URL} eigen && cd eigen \
        && git checkout ${BLD_EIGEN_VERSION} \
        && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/openrobots -DCMAKE_BUILD_TYPE=Release -DEIGEN_TEST_CXX11=ON .. \
        && make -j ${NCORE} blas && make -j ${NCORE} install && (cd blas && make install) # && rm -rf $RH_INSTALLDIR/eigen

# Python
RUN apt-get update \
    && apt-get install \
	python3 python3-dev python3-pip python3-pyqt5 \
        --assume-yes --no-install-recommends && apt-get clean && rm -rf /var/lib/apt/lists/*
# tzdata non-interactive
RUN ln -fs /usr/share/zoneinfo/Europe/London /etc/localtime \
  && apt-get update \
  && apt-get install -y tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Mujoco
RUN pip3 install meshcat mujoco matplotlib mediapy

