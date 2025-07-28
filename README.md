!!This repository is an adaptedversion of a project done by Axel Brunnbauer that was done within the ADEX project https://github.com/AutonomousDrivingExaminer!!

The original project implemented the CAT method for the Carla simulator.
This repository built on top of this and made the project work in a setup together with Autoware and also integrated other strategies than CAT to be able to make comparisons.

Below you can find the instructions to setup for the original Repository from Axel. Extensive instructions to run the complete setup with Autoware and the other methods can be found in a file called Documentation.pdf.

!!Also if you run the project using docker via ./scripts/run-docker-xosc_scenario.sh you dont need the manual setup steps from below.!!

# ADEX - CAT Example

## Installation

1. Download the pretrained DenseTNT model from [here](https://drive.google.com/drive/folders/1xVQ84pF5clVtKw6d4NCC-0mYbo4cIZ_a) and place it in `./cat/advgen/pretrained`. 
2. Install dependencies from `./cat`: `pip install -r cat/requirements.txt`
3. Compile Cython extension: `cd cat/advgen && cythonize -a -i utils_cython.pyx`
4. Install remaining dependencies: `pip install -r requirements.txt`

### Docker
To build the docker image, just run `make cat-docker`.

## Usage

### Docker
1. Start CARLA: `./scripts/start-carla.sh`
2. Start the container with the example script: `./scripts/run-docker-cat-example.sh`

### Locally
1. Start CARLA: `./scripts/start-carla.sh`
2. Start the python script: `PYTHONPATH=cat python cat_example.py`
