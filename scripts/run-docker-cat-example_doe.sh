#!/bin/bash
make cat-docker
docker run -v /var/run/docker.sock:/var/run/docker.sock -v /dev/pts:/dev/pts -it --rm --network host adex/cat:latest /bin/bash -c "SDL_VIDEODRIVER=dummy python3 activeDoe_test_scenic.py"
