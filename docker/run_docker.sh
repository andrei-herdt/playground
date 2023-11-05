#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
	echo "usage: $0 [-z|--zsh] [-l|--log-dir LOG_DIR_PATH] [-v EXTRA_MOUNT_DIR] [-p|--port PORT]"
	echo "    [-n|--name NAME] [-d|--device DEVICE] [-i|--image IMAGE] [-?|--help] [-c CMD]"
	echo "  -z, --zsh"
	echo "        Use z-shell instead of bash"
	echo "  -l, --log-dir"
	echo "        The location of the mounted NAS"
	echo "  -v    The location of an additional directory to be mounted in the docker container."
	echo "        May be specified multiple times."
	echo "        NOTE: the directory is mounted in the same location of the host."
	echo "  -p, --port"
	echo "        Specify a specific port rather than using host networking (not recommended)."
	echo "        May be specified multiple times."
	echo "  -n, --name"
	echo "        Name of the container."
	echo "  -i, --image"
	echo "        Specify the Docker image suffix (e.g., dev, emacs, ..., default is dev)"
	echo "  -d, --device"
	echo "        path to additional devices to map."
	echo "  -D, --detach"
	echo "        run docker container in detach mode."
	echo "  --ssh"
	echo "        Flag to set graphics card options for ssh connection"
	echo "  --entrypoint"
	echo "        Specify an alternative entry point (/bin/bash is a default)"
}

_SHELL=/bin/bash
LOG_DIR_MOUNT="/mnt/Robotics"
LOG_DIR=${LOG_DIR_MOUNT}
EXTRA_MOUNTS_=()
DEVICES_=()
SSH=0
NAME=''
NETWORKING=''
DOCKER_EXTRA_FLAGS_=()
ENTRYPOINT="/bin/tmux"
DEVIMAGE=_dev

while [ "$1" != "" ]; do
	case $1 in
	-z | --zsh)
		_SHELL=/usr/bin/zsh
		;;
	-l | --log-dir)
		shift
		LOG_DIR=$1
		;;
	-v)
		shift
		EXTRA_MOUNTS_+=("$1")
		;;
	-p | --port)
		shift
		NETWORKING="${NETWORKING} -p $1:$1"
		;;
	-n | --name)
		shift
		NAME='--name '$1
		;;
	-i | --image)
		shift
		DEVIMAGE=_$1
		;;
	-d | --device)
		shift
		DEVICES_+=("$1")
		;;
	-D | --detach)
		DOCKER_EXTRA_FLAGS_+=("-d")
		;;
	--ssh)
		SSH=1
		;;
	--entrypoint)
		shift
		ENTRYPOINT=$1
		;;
	-? | --help)
		usage
		exit
		;;
	*)
		usage
		exit
		;;
	esac
	shift
done

# Default to host-mode networking if no specific port was specified.
NETWORKING=${NETWORKING:-"--network host"}

source "${DIR}/docker_version.sh"

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

# A fix for the weird xauth issues
if [ -d $XAUTH ]; then
	rm -rf $XAUTH
	touch $XAUTH
fi

if [ ! -f $XAUTH ]; then
	touch $XAUTH
fi

xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
sudo chown -R "$(whoami)" $XSOCK $XAUTH
GPUOPT="--device=/dev/dri/card0:/dev/dri/card0"

if [ -f /usr/bin/nvidia-smi ]; then
	## this is needed to support optimus - we can have nvidia drivers set up,
	## but run an integrated GPU instead
	NVENV=$(glxinfo | grep "GL vendor string: NVIDIA")
	if [ "${NVENV}" != "" ]; then
		GPUOPT="--gpus all"
	fi
fi
if [ $SSH -eq 1 ]; then
	GPUOPT=''
fi

if [ -d "${LOG_DIR}" ]; then
	MOUNT_LOG_DIR="-v ${LOG_DIR}:${LOG_DIR_MOUNT}"
fi

EXTRA_MOUNTS=""
for m in "${EXTRA_MOUNTS_[@]}"; do
	if [ -d "${m}" ]; then
		EXTRA_MOUNTS="${EXTRA_MOUNTS} -v ${m}:${m}"
	else
		echo "Attempt to mount '${m}' but not a valid directory!"
		exit 1
	fi
done

if [ -d "/opt/priv" ]; then
	EXTRA_MOUNTS="${EXTRA_MOUNTS} -v /opt/priv:/opt/priv"
fi

if [ -d "${HOME}/.aws" ]; then
	EXTRA_MOUNTS="${EXTRA_MOUNTS} -v ${HOME}/.aws:/root/.aws"
fi

DEVICES=""
for d in "${DEVICES_[@]}"; do
	if [ -e "${d}" ]; then
		DEVICES="${DEVICES} --device=${d}"
	else
		echo "Attempt to connect device '${d}' but not a valid device!"
		exit 1
	fi
done

DOCKER_EXTRA_FLAGS=""
for f in "${DOCKER_EXTRA_FLAGS_[@]}"; do
	DOCKER_EXTRA_FLAGS="${DOCKER_EXTRA_FLAGS} ${f}"
done

# Add extra hosts:
if [ -f hosts ]; then
	EXTRA_HOSTS=""
	while read -r line; do
		hosts=("$line")
		for h in "${hosts[@]:1}"; do
			EXTRA_HOSTS="${EXTRA_HOSTS} --add-host=$h:${hosts[0]}"
		done
	done < <(grep -o '^[^#]*' hosts)
fi

xhost +local:docker

docker run ${NAME} --privileged --shm-size=512m --cap-add=SYS_PTRACE --security-opt seccomp=unconfined $GPUOPT \
	${NETWORKING} -ti \
	-v $XSOCK:$XSOCK \
	-v $XAUTH:$XAUTH \
	-v ${PWD}/..:/workdir/playground \
	-v ${PWD}/workdir:/workdir \
	-v ${PWD}/workdir/.config/nvim:/root/.config/nvim \
	-v ${PWD}/workdir/.local/share/nvim:/root/.local/share/nvim \
	-v ~/.bash_history:/root/.bash_history \
	-v ~/.zsh_history:/root/.zsh_history \
	-v ~/.ssh/:/root/.ssh/ \
	${MOUNT_LOG_DIR} \
	${EXTRA_MOUNTS} \
	${EXTRA_HOSTS} \
	${DEVICES} \
	${DOCKER_EXTRA_FLAGS} \
	--entrypoint ${ENTRYPOINT} \
	--workdir /workdir/playground \
	-e XAUTHORITY=$XAUTH \
	-e DISPLAY=${DISPLAY} \
	--runtime=nvidia \
	${IMAGE}${DEVIMAGE}:$VERSION
