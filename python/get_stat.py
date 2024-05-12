import os
import time
import docker
from apscheduler.schedulers.background import BackgroundScheduler


def calculate_cpu_percent(d):
    cpu_count = len(d["cpu_stats"]["cpu_usage"]["percpu_usage"])
    cpu_percent = 0.0
    cpu_delta = float(d["cpu_stats"]["cpu_usage"]["total_usage"]) - \
                float(d["precpu_stats"]["cpu_usage"]["total_usage"])
    system_delta = float(d["cpu_stats"]["system_cpu_usage"]) - \
                   float(d["precpu_stats"]["system_cpu_usage"])
    if system_delta > 0.0:
        cpu_percent = cpu_delta / system_delta * 100.0 * cpu_count
    return cpu_percent


def graceful_chain_get(d, *args, default=None):
    t = d
    for a in args:
        try:
            t = t[a]
        except (KeyError, ValueError, TypeError, AttributeError):
            return default
    return t


def calculate_blkio_bytes(d):
    """
    :param d:
    :return: (read_bytes, wrote_bytes), ints
    """
    bytes_stats = graceful_chain_get(d, "blkio_stats", "io_service_bytes_recursive")
    if not bytes_stats:
        return 0, 0
    r = 0
    w = 0
    for s in bytes_stats:
        if s["op"] == "Read":
            r += s["value"]
        elif s["op"] == "Write":
            w += s["value"]
    return r, w


def calculate_network_bytes(d):
    """
    :param d:
    :return: (received_bytes, transceived_bytes), ints
    """
    networks = graceful_chain_get(d, "networks")
    if not networks:
        return 0, 0
    r = 0
    t = 0
    for if_name, data in networks.items():
        r += data["rx_bytes"]
        t += data["tx_bytes"]
    return r, t


def calculate_memory_bytes(d):
    return d["memory_stats"]["usage"] - d["memory_stats"]["stats"]["cache"] + d["memory_stats"]["stats"]["active_file"]


class ResourceLogger:

    def __init__(self, num_client, container_name, log_file="resource.log"):
        self._num_client = num_client
        self._container_name = container_name
        self._records = []
        self._container_stats = []
        self._log_file = log_file
        self._client = docker.from_env()
        for i in range(self._num_client):
            self._records.append([0, 0, 0, 0, 0, 0])
            self._container_stats.append(self._try_connect(self._container_name.format(i)))
        self._check_all_container()
    
    def _try_connect(self, cname):
        try:
            container = self._client.containers.get(cname)
            response = container.stats(decode=True)
            next(response)
            return response
        except docker.errors.NotFound:
            return None

    def _check_all_container(self):
        self._all_container_ready = True
        for i in range(self._num_client):
            if self._container_stats[i] is None:
                self._all_container_ready = False
                print(self._container_name.format(i), "not ready")
                return False
        os.system("docker stats --no-stream " + " ".join([self._container_name.format(i) for i in range(self._num_client)]))
        return True
    
    def collect(self):
        if not self._all_container_ready:
            for i in range(self._num_client):
                if self._container_stats[i] is None:
                    self._container_stats[i] = self._try_connect(self._container_name.format(i))
            self._check_all_container()
        for i in range(self._num_client):
            if self._container_stats[i] is None:
                continue
            try:
                status_chack = next(self._container_stats[i])
                # print(status_chack)
                cpu_usage = calculate_cpu_percent(status_chack)
                blockIn, blockOut = calculate_blkio_bytes(status_chack)
                netIn, netOut = calculate_network_bytes(status_chack)
                memory = calculate_memory_bytes(status_chack)
                tmp_records = [cpu_usage, memory / 2**30, netIn / 2**30, netOut / 2**30, blockIn / 2**30, blockOut / 2**30]
                for j in range(6):
                    self._records[i][j] = max(self._records[i][j], tmp_records[j])
            except Exception as e:
                print(f"Cannot get containers' information! Try later...({e})")
                return None
        with open(self._log_file, 'w') as f:
            for i in range(self._num_client):
                f.write(f"{self._container_name.format(i)} " + str(self._records[i]) + '\n')

    def start(self, interval=2):
        self._scheduler = BackgroundScheduler(job_defaults={'max_instances': 1})
        self._scheduler.start()
        self._scheduler.add_job(
            self.collect, trigger='cron', second=f'*/{interval}',
            id='log_resources'
        )

    def stop(self):
        self._scheduler.remove_job('log_resources')


if __name__ == "__main__":
    rl = ResourceLogger(8, "DecFedSVD_C{}")
    rl.start(1)
    time.sleep(1000)
    