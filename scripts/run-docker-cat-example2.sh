#!/bin/bash
make cat-docker
docker run -it --rm --network host adex/cat:latest /bin/bash -c "SDL_VIDEODRIVER=dummy python3 cat_example_2.py"
