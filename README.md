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