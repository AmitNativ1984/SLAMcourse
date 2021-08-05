#!/bin/bash
echo " "
echo "usage: starts the Docker container"
echo "----------------------------------"


# generate mount commands
DOCKER_ROOT="/home"
DEV_VOLUME="/home/$USER/Dev:$DOCKER_ROOT/Dev"

CONTAINER_IMAGE="amitnativ/slam:latest"

# running docker
docker run -it --rm -d \
                    -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v $DEV_VOLUME \
                    -p 5000:5000 \
                    -p 8888:8888 \
                    -e DISPLAY=$DISPLAY \
                    $CONTAINER_IMAGE
