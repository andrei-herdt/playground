#!/bin/bash

NCORES=$(cat /proc/cpuinfo | grep processor | wc -l)
((--NCORES))
docker build --build-arg "NCORE=${NCORES}" --cpuset-cpus "0-${NCORES}" . -t play_image:1.0 -f Dockerfile   \
&& docker build --build-arg "NCORE=${NCORES}" --cpuset-cpus "0-${NCORES}" . -t play_image_dev:1.0 -f Dockerfile_dev
