#!/bin/bash

docker exec --user "docker_minkloc3d" -it ${USER}_minkloc3d \
    /bin/bash -c "cd /home/docker_minkloc3d; echo minkloc3d container; echo ; /bin/bash"