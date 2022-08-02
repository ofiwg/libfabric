

def efa_run_client_server_test(cmdline_args, executable, iteration_type,
                               completion_type, memory_type, message_size,
                               warmup_iteration_type=None):
    from common import ClientServerTest
    # It is observed that cuda tests requires larger time-out limit (~240 secs) to test all
    # message sizes for libfabric's debug and mem-poisoning builds, on p4d instances.
    timeout = None
    if "cuda" in memory_type and message_size == "all":
        timeout = 240

    test = ClientServerTest(cmdline_args, executable, iteration_type,
                            completion_type=completion_type,
                            datacheck_type="with_datacheck",
                            message_size=message_size,
                            memory_type=memory_type,
                            timeout=timeout,
                            warmup_iteration_type=warmup_iteration_type)
    test.run()

def efa_retrieve_hw_counter_value(hostname, hw_counter_name):
    import subprocess
    """
    retrieve the value of EFA's hardware counter
    hostname: a host that has efa
    hw_counter_name: EFA hardware counter name. Options are: lifespan, rdma_read_resp_bytes, rdma_read_wrs,recv_wrs,
                     rx_drops, send_bytes, tx_bytes, rdma_read_bytes,  rdma_read_wr_err, recv_bytes, rx_bytes, rx_pkts, send_wrs, tx_pkts
    return: an integer that is sum of all EFA device's counter
    """
    command = "ssh {} cat \"/sys/class/infiniband/*/ports/*/hw_counters/{}\"".format(hostname, hw_counter_name)
    try:
        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        # this can happen when OS is using older version of EFA kernel module
        return None

    linelist = process.stdout.split()
    sumvalue = 0
    for strvalue in linelist:
        sumvalue += int(strvalue)
    return sumvalue

def has_gdrcopy(hostname):
    import subprocess
    """
    determine whether a host has gdrcopy installed
    hostname: a host
    return: a boolean
    """
    command = "ssh {} /usr/sbin/lsmod | grep gdrdrv".format(hostname)
    process = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE)
    return process.returncode == 0
