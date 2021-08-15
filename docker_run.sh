#!/bin/bash
echo " "
echo "usage: starts the Docker container"
echo "----------------------------------"


# generate mount commands
DOCKER_ROOT="/home"
DEV_VOLUME="/home/$USER/Dev:$DOCKER_ROOT/Dev"

CONTAINER_IMAGE="amitnativ/slam:opencv-matplotlib"

# running docker
xhost +
docker run -it --rm -d \
                    --net=host \
                    -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v $DEV_VOLUME \
                    -e DISPLAY=$DISPLAY \
                    $CONTAINER_IMAGE
