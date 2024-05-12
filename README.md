# Excalibur

This is a fully functional Excalibur system for the paper "Efficient Decentralized Federated Singular Vector Decomposition", USENIX ATC'24.

## Reproduce the Results

### Prepare the environments

#### Step 1: OS and software.

The system is implemented on Linux OS (tested on Ubuntu 20.04) and the following software is needed to reproduce the results:

- Docker
  - [Installing instructions.](https://docs.docker.com/engine/install/ubuntu/)
  - Tested version: Docker version 20.10.21
- Docker-compose
  - [Installing instructions.](https://docs.docker.com/compose/install/linux/)
  - Tested version: docker-compose version 1.28.5
- MiniConda
  - [Installing instructions.](https://docs.anaconda.com/free/miniconda/miniconda-install/)
  - Tested version: conda 23.5.2

#### Step 2: Build the docker image and python env.

```bash
git clone https://github.com/Di-Chai/Excalibur
cd Excalibur/docker
docker build . -t excalibur:v1
conda create -y -n excalibur python=3.8
conda activate excalibur
pip install -r requirements.txt
```

#### Step 3: Download the datasets.



#### Step 4: Reproduce the results using scripts.



## Build the System (for debugging or reuse for other projects)

With docker:

```bash

mkdir build
docker run -it --rm -v $(pwd):/data -w /data/build excalibur:v1 bash -c "cmake .. && make"

```

Without docker (need Intel MKL installed):

```bash

mkdir build
source /opt/intel/oneapi/setvars.sh
cd build
cmake ..
make

```
