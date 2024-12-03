#!/bin/bash
docker run -it --rm --network host adex/cat:latest /bin/bash -c "SDL_VIDEODRIVER=dummy python cat_example_2.py"
