import subprocess
from common import SshConnectionError, is_ssh_connection_error, has_ssh_connection_err_msg
from retrying import retry

def efa_run_client_server_test(cmdline_args, executable, iteration_type,
                               completion_type, memory_type, message_size,
                               warmup_iteration_type=None):
    from common import ClientServerTest
    # It is observed that cuda tests requires larger time-out limit to test all
    # message sizes (especailly when running with multiple workers).
    timeout = None
    if "cuda" in memory_type:
        timeout = max(1000, cmdline_args.timeout)

    test = ClientServerTest(cmdline_args, executable, iteration_type,
                            completion_type=completion_type,
                            datacheck_type="with_datacheck",
                            message_size=message_size,
                            memory_type=memory_type,
                            timeout=timeout,
                            warmup_iteration_type=warmup_iteration_type)
    test.run()

@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def efa_retrieve_hw_counter_value(hostname, hw_counter_name):
    """
    retrieve the value of EFA's hardware counter
    hostname: a host that has efa
    hw_counter_name: EFA hardware counter name. Options are: lifespan, rdma_read_resp_bytes, rdma_read_wrs,recv_wrs,
                     rx_drops, send_bytes, tx_bytes, rdma_read_bytes,  rdma_read_wr_err, recv_bytes, rx_bytes, rx_pkts, send_wrs, tx_pkts
    return: an integer that is sum of all EFA device's counter
    """
    command = 'ssh {} cat "/sys/class/infiniband/*/ports/*/hw_counters/{}"'.format(hostname, hw_counter_name)
    process = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if process.returncode != 0:
        if process.stderr and has_ssh_connection_err_msg(process.stderr):
            print("encountered ssh connection issue")
            raise SshConnectionError()
        # this can happen when OS is using older version of EFA kernel module
        return None

    linelist = process.stdout.split()
    sumvalue = 0
    for strvalue in linelist:
        sumvalue += int(strvalue)
    return sumvalue

def has_gdrcopy(hostname):
    """
    determine whether a host has gdrcopy installed
    hostname: a host
    return: a boolean
    """
    command = "ssh {} /usr/sbin/lsmod | grep gdrdrv".format(hostname)
    process = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE)
    return process.returncode == 0

def efa_retrieve_gid(hostname):
    """
    return the GID of efa device on a host
    hostname: a host
    return: a string if the host has efa device,
            None otherwise
    """
    command = "ssh {} ibv_devinfo  -v | grep GID | awk '{{print $NF}}' | head -n 1".format(hostname)
    try:
        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        # this can happen on instance without EFA device
        return None

    return process.stdout.decode("utf-8").strip()
