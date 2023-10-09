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
#ARG BLD_MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
ARG BLD_EIGENPY_URL=https://github.com/stack-of-tasks/eigenpy.git
ARG BLD_GIT_LFS=https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz

FROM ubuntu:22.04
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

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
ARG BLD_GIT_LFS

ENV INSTALLDIR=/root

RUN cat /etc/resolv.conf

RUN apt-get update \
    && apt-get install \
    sudo \
    git \
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
RUN cd /root && wget --quiet ${BLD_CMAKE_URL} && tar -xf cmake-${BLD_CMAKE_VERSION}.tar.gz \
   && cd /root/cmake-${BLD_CMAKE_VERSION} && ./bootstrap && make -j5 && make -j4 install && cd /root && rm -rf /root/cmake-${BLD_CMAKE_VERSION}

# Anaconda 3
#RUN wget --quiet ${BLD_MINICONDA_URL} -O ~/anaconda.sh && \
#    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh && \
#    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate base" >> ~/.bashrc

# scientific python packages 
#ENV PATH="/opt/conda/bin:$PATH"
#ENV CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/opt/conda/include/python3.9"
#RUN . ~/.bashrc && conda activate base && conda install -y numpy scipy pybind11

# Install boost
RUN cd /root && wget -q -c ${BLD_BOOST_URL} -O /root/boost.tar.bz2 \
        && mkdir boost && cd /root/boost && tar -xf /root/boost.tar.bz2 --strip-components=1 && \
     cd /root/boost && \
          ./bootstrap.sh && \
           ./b2 -j8 install ; cd /root; rm -rf /root/boost*

# Eigen3
RUN cd $RH_INSTALLDIR \
        && git clone ${BLD_EIGEN_URL} eigen && cd eigen \
        && git checkout ${BLD_EIGEN_VERSION} \
        && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/openrobots -DCMAKE_BUILD_TYPE=Release -DEIGEN_TEST_CXX11=ON .. \
        && make -j8 blas && make -j8 install && (cd blas && make install) # && rm -rf $RH_INSTALLDIR/eigen

# Pinocchio
#RUN . ~/.bashrc && conda activate base && conda install -c conda-forge pinocchio tsid
RUN apt-get update \
    && apt-get install \
	python3 python3-dev python3-pip \
        --assume-yes --no-install-recommends && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip3 install meshcat mujoco matplotlib mediapy


# tzdata non-interactive
RUN ln -fs /usr/share/zoneinfo/Europe/London /etc/localtime \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN cd /root && wget --quiet ${BLD_GIT_LFS} -O /root/git-lfs.tar.bz2 \
        && mkdir git-lfs && cd /root/git-lfs && tar -xf /root/git-lfs.tar.bz2 \
        && ./install.sh && cd /root; rm -rf /root/git-lfs

# RUN cd $RH_INSTALLDIR && git clone ${BLD_URDFDOM_HEADERS_URL} urdfdom_headers && cd urdfdom_headers \
#     && git checkout ${BLD_URDFDOM_HEADERS_VERSION} && git submodule update --init --recursive \
#     && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release .. && make -j8 install \
#     && cd $RH_INSTALLDIR && rm -rf urdfdom_headers

# RUN cd $RH_INSTALLDIR && git -c http.sslVerify=false  clone ${BLD_TINYXML_URL} tinyxml && cd tinyxml\
#     && make all && make install \
#     && cd /root && rm -rf tinyxml

# RUN cd $RH_INSTALLDIR && git clone ${BLD_URDFDOM_URL} urdfdom && cd urdfdom \
#     && git checkout ${BLD_URDFDOM_VERSION} && git submodule update --init --recursive \
#     && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release .. && make -j8 install \
#     && cd $RH_INSTALLDIR && rm -rf urdfdom

# RUN cd $RH_INSTALLDIR \
#         && git clone ${BLD_EIGENPY_URL} eigenpy \
#         && cd eigenpy && git checkout ${BLD_EIGENPY_VERSION} && git submodule update --init --recursive \
#         && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/opt/openrobots -DCMAKE_BUILD_TYPE=Release .. && make -j8 install \
#         && cd $RH_INSTALLDIR # && rm -rf $RH_INSTALLDIR/eigenpy

# RUN cd $INSTALLDIR \
#      && git clone ${BLD_PINOCCHIO_URL} pinocchio \
#      && cd pinocchio && git checkout ${BLD_PINOCCHIO_VERSION} && git submodule update --init --recursive \
#      && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_WITH_URDF_SUPPORT=OFF .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=ON -DCMAKE_INSTALL_PREFIX=/opt/openrobots \
#      && make -j8 && make install
