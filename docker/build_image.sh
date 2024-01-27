#!/bin/bash

NCORES=$(cat /proc/cpuinfo | grep processor | wc -l)
((--NCORES))

BASE_IMAGE=ubuntu:22.04
if [ -f /usr/bin/nvidia-smi ]; then
	BASE_IMAGE=nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04
	echo "nvidia"
fi

docker build --build-arg "NCORE=${NCORES}" --build-arg "BASE_IMAGE=${BASE_IMAGE}" --cpuset-cpus "0-${NCORES}" . -t play_image:1.0 -f Dockerfile

if [ -f /usr/bin/nvidia-smi ]; then
	docker build --build-arg "NCORE=${NCORES}" --build-arg "BASE_IMAGE=${BASE_IMAGE}" --cpuset-cpus "0-${NCORES}" . -t play_image:1.0 -f Dockerfile_cuda
	echo "nvidia"
fi
docker build --build-arg "NCORE=${NCORES}" --cpuset-cpus "0-${NCORES}" . -t play_image_dev:1.0 -f Dockerfile_dev
