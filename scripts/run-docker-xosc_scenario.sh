#!/bin/bash
make cat-docker
docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /dev/pts:/dev/pts \
    -v /work/Valentin_dev/mats_traffic_gen_modifed:/workspace \
    -it --network host adex/cat:latest \
    /bin/bash -c "SDL_VIDEODRIVER=dummy python3 run_xosc_scenario.py"
