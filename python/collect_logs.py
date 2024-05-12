import os
import re
import sys
import pdb
import numpy as np


def parse_resource_usage(resource_log_filename):
    """
    [CPU, Memory, netIn, netOut, blockIn, blockOut] GB
    """
    with open(resource_log_filename) as f:
        resource_log = f.readlines()
    usage = [re.match(".*(\[.*\]).*", e)[1] for e in resource_log]
    usage = np.array([eval(e) for e in usage])
    return np.average(usage, axis=0).tolist()


def parse_clint_log(client_log_filename):
    with open(client_log_filename) as f:
        logs = f.readlines()
    patterns = [
        ".*TopK = (.*)",
        ".*Local N = (.*)",
        ".*is_memmap = (.*)",
        ".*evaluate = (.*)",
        ".*DecFedSVD totally Costs (.*)",
        ".*Final U OrthTest (.*)",
        ".*Final V OrthTest (.*)",
        ".*Final SVD Error (.*)",
        ".*Projection Distance U (.*)",
        ".*Projection Distance VT (.*)",
        ".*LR MSE Error (.*)",
    ]
    results = [-1] * len(patterns)
    for i in range(len(patterns)):
        for ll in logs:
            rmatch = re.match(patterns[i], ll)
            if rmatch:
                results[i] = rmatch.group(1)
    return results


if __name__ == "__main__":

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
        print(f"Reading log from {log_dir}")
    else:
        print(f"Please provide log_dir")
        exit(-1)
    
    log_records = [e for e in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, e))]
    
    result_file = os.path.join(log_dir, "results.csv")
    f = open(result_file, 'w')
    f.write("datasets, m, n, task, top_k, num_clients, seed, is_memmap, evaluate, bandwidth, latency, time, u_orth, v_orth, svd_error, u_pd, v_pd, lr_mse, CPU, Memory, netIn, netOut, blockIn, blockOut\n")

    for log_record in log_records:

        dataset, m, n, task, num_clients, seed, bandwidth, latency = log_record.split("_")[3:]
        
        log_path = os.path.join(log_dir, log_record)

        resource_log = os.path.join(log_path, "resource.log")
        CPU, Memory, netIn, netOut, blockIn, blockOut = parse_resource_usage(resource_log)

        client_logs = [
            os.path.join(log_path, e) 
            for e in os.listdir(log_path) 
            if e.startswith("Client") and e.endswith(".log")
        ]
        client_logs = sorted(client_logs, key=lambda x: int(x.split("/")[-1].strip("Client").strip(".log")))
        client_logs = [parse_clint_log(e) for e in client_logs]
        
        top_k = client_logs[0][0]
        local_ns = [int(e[1]) for e in client_logs]
        is_memmap = client_logs[0][2]
        evaluete = client_logs[0][3]
        total_time = np.mean([float(e[4]) for e in client_logs])
        u_orth_test = client_logs[0][5]
        v_orth_test = sum([float(client_logs[i][6]) * local_ns[i] for i in range(len(client_logs))]) / sum(local_ns)

        svd_error = sum([float(client_logs[i][7]) * local_ns[i] for i in range(len(client_logs))]) / sum(local_ns)
        u_pd = client_logs[0][8]
        v_pd = sum([float(client_logs[i][9]) * local_ns[i] for i in range(len(client_logs))]) / sum(local_ns)
        lr_error = [e[10] for e in client_logs if e[10] != -1]
        if len(lr_error) == 0:
            lr_error = -1
        else:
            lr_error = lr_error[0]

        data_records = f"{dataset}, {m}, {n}, {task}, {top_k}, {num_clients}, {seed}, {is_memmap}, {evaluete}, {bandwidth}, {latency}, {total_time}, {u_orth_test}, {v_orth_test}, {svd_error}, {u_pd}, {v_pd}, {lr_error}, {CPU}, {Memory}, {netIn}, {netOut}, {blockIn}, {blockOut}\n"
        f.write(data_records)
    
    f.close()

    print(f"Result saved to {result_file}")
    