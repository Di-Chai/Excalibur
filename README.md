# Excalibur

This is a fully functional Excalibur system for the paper "Efficient Decentralized Federated Singular Vector Decomposition", USENIX ATC'24.

## Project Structure

```bash
.
├── CMakeLists.txt
├── docker
│   ├── Dockerfile
│   └── requirements.txt
├── generate_data.cpp      // Scripts for generating the data.
├── include
│   ├── base.hpp
│   ├── bbtsvd.hpp
│   ├── client.hpp
│   ├── easylogging++.h
│   └── utils.hpp
├── main.cpp               // The main file executed by each peer
├── python
│   ├── collect_logs.py    // Collecting evaluation results.
│   ├── get_stat.py        // Monitering the container status.
├── README.md
├── src
│   ├── bbtsvd.cpp         // Matrix computation functions.
│   ├── client.cpp         // Communication and workflows.
│   ├── easylogging++.cc   // Only for logging.
│   └── utils.cpp          // Util functions, e.g., the evaluation metrics.
└── trial.py               // Scripts for reproducing the results.
```

## Prepare the environments

#### Step 1: OS and software.

The system is implemented on Linux OS (tested on Ubuntu 20.04) and the following software is needed to reproduce the results:

- Docker
  - [Installing instructions.](https://docs.docker.com/engine/install/ubuntu/)
  - Please also ensure the "docker" command can be executed without sudo. 
  - Tested version: Docker version 20.10.21
- Docker-compose
  - [Installing instructions.](https://docs.docker.com/compose/install/linux/)
  - Tested version: docker-compose version 1.28.5
- MiniConda
  - [Installing instructions.](https://docs.anaconda.com/free/miniconda/miniconda-install/)
  - Tested version: conda 23.5.2

#### Step 2: Build the docker image and create the python env.

```bash
git clone https://github.com/Di-Chai/Excalibur
cd Excalibur
docker build docker/ -t excalibur:v1
conda create -y -n excalibur python=3.8
conda activate excalibur
pip install -r docker/requirements.txt
```

```bash
# Env: conda activate excalibur.
# This will test SVD and its three applications using synthetic data.
python trial.py test
```

## Reproduce the Results

#### Step 1: Download the datasets.



#### Step 2: Reproduce the results using scripts.

(Note: The scripts use the docker-bridge network with subnet 171.20.0.0/24. If you already have a docker network with a conflict subnet, please remove it before running the scripts.)

Reproducing the accuracy results, i.e., Table 3.

```bash
python trial.py accuracy
```

Reproducing the efficiency results on SVD tasks, i.e., Figure 8.
```bash
python trial.py large_scale_svd_short_wide
python trial.py large_scale_svd_tall_skinny
python trial.py bandwidth
python trial.py latency
```

Reproducing the comparison to the HE-based method on the PCA application, i.e., Table 4.
```bash
python trial.py comparing_to_sfpca
```

Reproducing the comparison results on LR application, i.e., Figure 10.
```bash
python trial.py large_scale_lr_tall_skinny
```

Reproducing the scalability results, i.e., Figure 11.
```bash
python trial.py vary_num_clients_ts
```

Reproducing the results with and without the proposed optimizations, i.e., Figure 12.
```bash
python trial.py opt
```

## Build the System (for debugging or reuse for other projects)

With docker:

```bash
mkdir build
docker run -it --rm -v $(pwd):/data -w /data/build excalibur:v1 bash -c "cmake .. && make"
```

Without docker (need Intel MKL installed, [install instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)):

```bash
mkdir build
source /opt/intel/oneapi/setvars.sh
cd build
cmake ..
make
```
