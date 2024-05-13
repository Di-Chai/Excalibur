import os
import sys
import copy
import yaml
import json
import datetime
import platform
from python.get_stat import *

# Hardware config
project_path = os.path.dirname(os.path.abspath(__file__))
max_cpu_count = os.cpu_count()
core_per_peer = 4
memory_limit = "64GB"

# Rebuild
os.system(f'docker run -it --rm -v {project_path}:/data -w /data/build excalibur:v1 bash -c "cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx .. && make"')

# Dir Check
if not os.path.isdir("datasets"):
    os.makedirs("datasets")
if not os.path.isdir("logs"):
    os.makedirs("logs")

data_ids = {
    'wine': [0, 6497, 11, "/data/datasets/wine"],
    'mnist': [1, 70000, 784, "/data/datasets/mnist"],
    'ml100k': [2, 943, 500, "/data/datasets/ml100k"],
    'ml25m': [3, 62423, 162541, "/data/datasets/ml25m"],
    'synthetic': [4, None, None, "/data/datasets/synthetic"],
    'syntheticlr': [5, None, None, "/data/datasets/syntheticlr"],
    'synthetic_very_large': [6, None, None, "/data/datasets/synthetic_very_large"],
}
image = "excalibur:v1"
network_config = "tc qdisc add dev eth0 handle 1: root htb default 11 &&"\
    "tc class add dev eth0 parent 1: classid 1:1 htb rate 100000Mbit && "\
    "tc class add dev eth0 parent 1:1 classid 1:11 htb rate {} && "\
    "tc qdisc add dev eth0 parent 1:11 handle 10: netem delay {} &&"
main_commend = './main -l 171.20.0.2 -p 8000 -m {} -n {} -d {} -k {} -o {} -t {} -c {} -g {} -e {} -f {} -s {} -a {} && echo Wait 3 seconds for Container Status Collection && sleep 3'

default_config = {
    "networks": {"dec_fedsvd": {"driver": "bridge", "ipam": {"config": [{"subnet": "171.20.0.0/24"}]}}},
    "services": {"c1": {"cap_add": ["NET_ADMIN"], "networks": {"dec_fedsvd": {"ipv4_address": "171.20.0.2"}}, "working_dir": "/data/build"}},
    "version": "2"
}

svd_mode_name = ["svd", "lsa", "lpca", "rpca", "lr"]

"""
    /*
    svd_mode = 0, Full SVD
    svd_mode = 1, Two Side LSA
    svd_mode = 2, Left Side PCA
    svd_mode = 3, Right Side PCA
    svd_mode = 4, Linear Regression
    */
"""

# Windows platform does not support "tc" commend (including WSL 1/2)
limit_network = platform.system() == "Linux"

# Force re-generate the data and run all experiments (even history records exist)
force_run = True


def generate_data(dataset, seed, num_clients, svd_mode, m=None, n=None, evaluate=1):
    gd_commend = f'docker run -it --rm --name decfedsvd_gd -v {project_path}:/data -w /data/build excalibur:v1 ./gd {seed} {data_ids[dataset][0]} {num_clients} {data_ids[dataset][1] or m} {data_ids[dataset][2] or n} {evaluate} {svd_mode}'
    commend_file = os.path.join("datasets", dataset, "command.txt")
    if os.path.isdir(os.path.join("datasets", dataset)):
        client_files = [e for e in os.listdir(os.path.join("datasets", dataset)) if e.startswith("Client") and e.endswith(".mat")]
    else:
        os.makedirs(os.path.join("datasets", dataset))
        client_files = []
    all_file_exist = True
    for i in range(num_clients):
        if f"Client{i}.mat" not in client_files:
            all_file_exist = False
            break
    if all_file_exist and os.path.isfile(commend_file) and not force_run:
        with open(commend_file, "r") as f:
            recorded_commend = f.read()
        if recorded_commend == gd_commend:
            print("History Data Found!")
            return None
    os.system(gd_commend)
    print(gd_commend)
    with open(commend_file, "w") as f:
        f.write(gd_commend)


def run(dataset, m, n, num_clients, top_k, svd_mode, seed, bandwidth, delay, evaluate, is_memmap, opt_control=3):

    m = data_ids[dataset][1] or m
    n = data_ids[dataset][2] or n

    if num_clients * core_per_peer > max_cpu_count:
        raise ValueError(f"This machine does not have enough CPU cores to run the experiment! Require num_clients*core_per_peer={num_clients * core_per_peer} while max_cpu_count={max_cpu_count}")

    if dataset == "synthetic_very_large" and evaluate:
        raise ValueError("Cannot evaluate under synthetic_very_large")

    if svd_mode == 4 and dataset == "synthetic":
        dataset = "syntheticlr"

    log_dir = f"logs/{datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')}_{dataset}_{m}_{n}_{svd_mode_name[svd_mode]}_{num_clients}_{seed}_{bandwidth}_{delay}"
    history_logs = ["_".join(e.split("_")[-8:]) for e in os.listdir("logs")]
    if "_".join(log_dir.split("_")[-8:]) in history_logs and not force_run:
        print(f"Log exists and Skip the experiment : {log_dir}")
        return None
    
    generate_data(dataset=dataset, seed=seed, num_clients=num_clients, svd_mode=svd_mode, m=m, n=n, evaluate=evaluate)

    if svd_mode == 4:
        n += 1
        if dataset not in ("wine", "mnist", "ml100k", "syntheticlr"):
            print(dataset, "does not support LR task")
            return -1

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    container_log_dir = '/data/' + log_dir

    data_partions = [int(n / num_clients)] * num_clients
    for i in range(n % num_clients):
        data_partions[i] += 1

    for i in range(1, num_clients):
        default_config["services"][f'c{i+1}'] = copy.deepcopy(default_config["services"]['c1'])

    for i in range(num_clients):
        mc = main_commend.format(
            m, data_partions[i], data_ids[dataset][3], 
            top_k, svd_mode, num_clients, i, container_log_dir, evaluate, is_memmap, seed, opt_control
        )
        if limit_network:
            default_config['services'][f'c{i+1}']['command'] = f'sh -c "{network_config.format(bandwidth, delay)} {mc}"'
        else:
            default_config['services'][f'c{i+1}']['command'] = f'sh -c "{mc}"'

        default_config['services'][f'c{i+1}']["container_name"] = f"DecFedSVD_C{i}"
        default_config['services'][f'c{i+1}']["image"] = image
        default_config['services'][f'c{i+1}']["cpuset"] = f"{i*core_per_peer}-{(i+1)*core_per_peer-1}"
        default_config['services'][f'c{i+1}']["mem_limit"] = memory_limit
        default_config['services'][f'c{i+1}']["networks"]["dec_fedsvd"]["ipv4_address"] = f"171.20.0.{i+2}"
        default_config['services'][f'c{i+1}']["volumes"] = [f"{project_path}:/data"]

    with open('docker-compose.yml', 'w') as f:
        yaml.dump(default_config, f)

    if limit_network:
        rl = ResourceLogger(num_clients, "DecFedSVD_C{}", log_dir + '/resource.log')
        rl.start(1)
        time.sleep(3)
    os.system('docker-compose up')
    if limit_network:
        rl.stop()
    os.system("docker-compose rm -f")
    os.system("docker network rm decfedsvd_dec_fedsvd")


if len(sys.argv) > 1:
    benchmark = sys.argv[1]
    print(f"Benchmarking {benchmark}")
else:
    benchmark = "test"
    print(f"Running test")


bandwidth_benchmark = ['10000Mbit', '1000Mbit', '100Mbit', '10Mbit']
# RTT = 2 * latency
latency_benchmark = ['0ms', '5ms', '10ms', '15ms', '20ms', '25ms', '30ms', '35ms', '40ms', '45ms', '50ms']

large_scale_size = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 20000000, 30000000, 40000000, 50000000]

if benchmark == "debug":
    m = 1000
    n = 20000
    num_clients = 2
    evaluate = 1
    is_memmap = 0
    bandwidth = "10024Mbit"
    delay = "0ms"
    seed = 0
    dataset = "synthetic"
    top_k = -1
    svd_mode = 0
    opt_control = 3
    run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap, opt_control=opt_control)

if benchmark == "opt":
    n = 10000
    num_clients = 2
    evaluate = 0
    is_memmap = 0
    bandwidth = "1024Mbit"
    delay = "25ms"
    seed = 0
    dataset = "synthetic"
    top_k = -1
    svd_mode = 0
    for m in [20000, 40000, 60000, 80000, 100000]:
        for opt_control in [0, 1, 3]:
            run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap, opt_control=opt_control)

if benchmark == "test":
    num_clients = 2
    m = 10000; n = 1000
    evaluate = 1
    is_memmap = 0
    bandwidth = "10000Mbit"
    delay = "0ms"
    seed = 0
    dataset = "synthetic"
    for svd_mode in [0, 1, 2, 3, 4]:
        if svd_mode in [1, 2, 3]:
            if (data_ids[dataset][2] or n) < 100:
                top_k = 4
            else:
                top_k = 16
        else:
            top_k = -1
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "accuracy":
    # Accuracy Test
    num_clients = 2
    m = 10000; n = 1000
    evaluate = 1; is_memmap = 0
    bandwidth = "10000Mbit"
    delay = "0ms"
    seed = 0
    for dataset in ["wine", "ml100k", "mnist", "synthetic"]:
        for svd_mode in [0]:
            if svd_mode in [1, 2, 3]:
                if (data_ids[dataset][2] or n) < 100:
                    top_k = 4
                else:
                    top_k = 16
            else:
                top_k = -1
            run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "bandwidth":
    dataset = "synthetic"
    num_clients = 2
    delay = "0ms"
    evaluate = 0
    is_memmap = 0
    svd_mode = 0
    top_k = -1
    seed = 0
    
    m = 1000000
    n = 1000
    for bandwidth in bandwidth_benchmark:
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

    m = 1000
    n = 1000000
    for bandwidth in bandwidth_benchmark:
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "latency":
    dataset = "synthetic"
    num_clients = 2
    bandwidth = "10000Mbit"
    evaluate = 0
    is_memmap = 0
    svd_mode = 0
    top_k = -1
    seed = 0
    
    m = 1000000
    n = 1000
    for delay in latency_benchmark:
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

    m = 1000
    n = 1000000
    for delay in latency_benchmark:
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "large_scale_svd_tall_skinny":
    dataset = "synthetic"
    num_clients = 2
    bandwidth = "1024Mbit"
    delay = "25ms"
    evaluate = 0
    is_memmap = 1
    seed = 0
    svd_mode = 0
    top_k = -1
    n = 1000
    for m in large_scale_size:
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "large_scale_lr_tall_skinny":
    dataset = "synthetic"
    num_clients = 2
    bandwidth = "1024Mbit"
    delay = "25ms"
    evaluate = 0
    is_memmap = 1
    seed = 0
    svd_mode = 4
    top_k = -1
    n = 1000
    for m in large_scale_size:
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "large_scale_svd_short_wide":
    dataset = "synthetic"
    num_clients = 2
    bandwidth = "1024Mbit"
    delay = "25ms"
    evaluate = 0
    is_memmap = 1
    seed = 0
    svd_mode = 0
    top_k = -1
    m = 1000
    for n in large_scale_size:
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "large_scale_lpca_short_wide":
    dataset = "synthetic"
    num_clients = 2
    bandwidth = "1024Mbit"
    delay = "25ms"
    evaluate = 0
    is_memmap = 1
    seed = 0
    svd_mode = 2
    top_k = 5
    m = 1000
    for n in large_scale_size:
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "large_scale_rpca":
    dataset = "synthetic_very_large"
    num_clients = 2
    bandwidth = "1024Mbit"
    delay = "25ms"
    evaluate = 0
    is_memmap = 1
    seed = 0
    svd_mode = 3
    top_k = 5
    m = 1000000
    n = 50000
    run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "large_scale_lsa":
    dataset = "ml25m"
    num_clients = 2
    bandwidth = "1024Mbit"
    delay = "25ms"
    evaluate = 0
    is_memmap = 1
    seed = 0
    svd_mode = 1
    top_k = 256
    m = -1
    n = -1
    run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "vary_num_clients_sw":
    m = 1000
    ni = 100000
    evaluate = 0
    is_memmap = 0
    bandwidth = "1000Mbit"
    delay = "25ms"
    seed = 0
    dataset = "synthetic"
    for num_clients in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        n = num_clients * ni
        for svd_mode in [0]:
            if svd_mode in [1, 2, 3]:
                if (data_ids[dataset][2] or n) < 100:
                    top_k = 4
                else:
                    top_k = 16
            else:
                top_k = -1
            run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "vary_num_clients_ts":
    num_clients = 2
    m = 100000
    ni = 250
    evaluate = 0
    is_memmap = 0
    bandwidth = "1024Mbit"
    delay = "25ms"
    seed = 0
    dataset = "synthetic"
    for num_clients in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        n = num_clients * ni
        for svd_mode in [0, 1]:
            if svd_mode in [1, 2, 3]:
                if (data_ids[dataset][2] or n) < 100:
                    top_k = 4
                else:
                    top_k = 16
            else:
                top_k = -1
            run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "vary_num_clients_fix_size":
    num_clients = 2
    m = 1000
    n = 1000000
    evaluate = 0
    is_memmap = 0
    bandwidth = "1000Mbit"
    delay = "25ms"
    seed = 0
    dataset = "synthetic"
    for num_clients in [2, 4, 8, 16, 32]:
        for svd_mode in [0]:
            if svd_mode in [1, 2, 3]:
                if (data_ids[dataset][2] or n) < 100:
                    top_k = 4
                else:
                    top_k = 16
            else:
                top_k = -1
            run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "comparing_to_sfpca":
    dataset = 'synthetic'
    num_clients = 6
    bandwidth = "1024Mbit"
    delay = "10ms"
    evaluate = 0
    is_memmap = 0
    seed = 0
    svd_mode = 2
    top_k = 5
    m = 1000
    n = 10000000
    run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)

if benchmark == "vary_min_mn":
    num_clients = 2
    evaluate = 0
    is_memmap = 0
    bandwidth = "1024Mbit"
    delay = "10ms"
    seed = 0
    dataset = "synthetic"
    svd_mode = 0
    top_k = -1
    for m in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000]:
        n = m
        run(dataset=dataset, m=m, n=n, top_k=top_k, num_clients=num_clients, svd_mode=svd_mode, seed=seed, bandwidth=bandwidth, delay=delay, evaluate=evaluate, is_memmap=is_memmap)
