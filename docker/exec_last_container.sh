#!/bin/bash

LAST_ID=$(docker container ls -q -l)
docker exec -w /workdir/playground -it ${LAST_ID} /bin/tmux
