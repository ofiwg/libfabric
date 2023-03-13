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
def efa_retrieve_hw_counter_value(hostname, hw_counter_name, efa_device_name=None):
    """
    retrieve the value of EFA's hardware counter
    hostname: a host that has efa
    hw_counter_name: EFA hardware counter name. Options are: lifespan, rdma_read_resp_bytes, rdma_read_wrs,recv_wrs,
                     rx_drops, send_bytes, tx_bytes, rdma_read_bytes,  rdma_read_wr_err, recv_bytes, rx_bytes, rx_pkts, send_wrs, tx_pkts
    efa_device_name: Name of the EFA device. Corresponds to the name of the EFA device's directory
    return: an integer that is sum of all EFA device's counter
    """

    if efa_device_name:
        efa_device_dir = efa_device_name
    else:
        efa_device_dir = '*'

    command = 'ssh {} cat "/sys/class/infiniband/{}/ports/*/hw_counters/{}"'.format(hostname, efa_device_dir, hw_counter_name)
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
    command = "ssh {} /bin/bash --login -c lsmod | grep gdrdrv".format(hostname)
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

@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def get_efa_domain_names(server_id):
    timeout = 60
    process_timed_out = False

    # This command returns a list of EFA domain names and its related info
    command = "ssh {} fi_info -p efa".format(server_id)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
 
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.terminate()
        process_timed_out = True

    assert not process_timed_out, "Process timed out"
    
    errors = p.stderr.readlines()
    for error in errors:
        error = error.strip()
        if "fi_getinfo: -61" in error:
            raise Exception("No EFA devices/domain names found")

        if has_ssh_connection_err_msg(error):
            raise SshConnectionError()

    efa_domain_names = []
    for line in p.stdout:
        line = line.strip()
        if 'domain' in line:
            domain_name = line.split(': ')[1]
            efa_domain_names.append(domain_name)

    return efa_domain_names

def get_efa_device_names(server_id):
    domain_names = get_efa_domain_names(server_id)
    device_names = set()
    for domain_name in domain_names:
        itemlist = domain_name.split("-")
        assert len(itemlist) == 2
        assert itemlist[1] in ["rdm", "dgrm"]
        device_names.add(itemlist[0])
    return list(device_names)
