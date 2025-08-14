import os
import subprocess
import functools
import pytest
from enum import IntEnum
from common import SshConnectionError, is_ssh_connection_error, has_ssh_connection_err_msg, ClientServerTest
from retrying import retry


class CudaMemorySupport(IntEnum):
    NOT_INITIALIZED = -1
    NOT_SUPPORTED = 0
    DMA_BUF_ONLY = 1
    GDR_ONLY = 2
    DMABUF_GDR_BOTH = 3

    def __str__(self):
        return self.name

@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def get_cuda_memory_support(cmdline_args, ip):
    """
    Execute check_dmabuf binary to determine CUDA memory support capabilities.
    
    Args:
        cmdline_args: Command line arguments containing binpath, timeout, provider, and environments.
        ip: IP address or hostname of the target machine.
        
    Returns:
        CudaMemorySupport: Enum value indicating hardware CUDA memory support type.
        
    Notes:
        - Executes check_dmabuf binary remotely via SSH with timeout
        - Maps return code directly to CudaMemorySupport enum values
        - Returns UNKNOWN for negative return codes indicating errors
        - Retries on SSH connection errors up to 3 times
    """
    binpath = cmdline_args.binpath or ""
    cmd = "timeout " + str(cmdline_args.timeout) \
          + " " + os.path.join(binpath, "check_cuda_dmabuf") \
          + " -p " + cmdline_args.provider
    if cmdline_args.environments:
        cmd = cmdline_args.environments + " " + cmd
    
    proc = subprocess.run("ssh {} {}".format(ip, cmd),
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT,
               shell=True,
               universal_newlines=True)
    
    if has_ssh_connection_err_msg(proc.stdout):
        raise SshConnectionError()
    
    if proc.returncode < 0:
        return CudaMemorySupport.NOT_SUPPORTED
    
    print(f"The ssh return is {proc}")
    rc = proc.returncode
    if rc not in (CudaMemorySupport.NOT_SUPPORTED,
                  CudaMemorySupport.DMA_BUF_ONLY,
                  CudaMemorySupport.GDR_ONLY,
                  CudaMemorySupport.DMABUF_GDR_BOTH):
        print(f"[warn] check_dmabuf returned unexpected code {rc}, treating as NOT_INITIALIZED")
        return CudaMemorySupport.NOT_INITIALIZED

    return CudaMemorySupport(rc)

def efa_run_client_server_test(cmdline_args, executable, iteration_type,
                               completion_semantic, memory_type, message_size,
                               warmup_iteration_type=None, timeout=None,
                               completion_type="queue", fabric=None,
                               additional_env=''):
    if timeout is None:
        timeout = cmdline_args.timeout

    # It is observed that cuda tests requires larger time-out limit to test all
    # message sizes (especailly when running with multiple workers).
    if "cuda" in memory_type:
        timeout = max(1000, timeout)

    test = ClientServerTest(cmdline_args, executable, iteration_type,
                            completion_semantic=completion_semantic,
                            datacheck_type="with_datacheck",
                            message_size=message_size,
                            memory_type=memory_type,
                            timeout=timeout,
                            warmup_iteration_type=warmup_iteration_type,
                            completion_type=completion_type, fabric=fabric,
                            additional_env=additional_env)
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

@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def has_rdma(cmdline_args, operation):
    """
    determine whether both client/server sides have rdma <operation> enabled in efa device
    cmdline_args: command line argument object, including server/client id, and timeout
    operation: rdma operation name, allowed values are read and write
    return: a boolean
    """
    assert operation in ["read", "write", "writedata"]
    binpath = cmdline_args.binpath or ""
    cmd = "timeout " + str(cmdline_args.timeout) \
          + " " + os.path.join(binpath, f"fi_efa_rdma_checker -o {operation}")
    if cmdline_args.environments:
        cmd = cmdline_args.environments + " " + cmd
    server_proc = subprocess.run("ssh {} {}".format(cmdline_args.server_id, cmd),
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT,
               shell=True,
               universal_newlines=True)
    if has_ssh_connection_err_msg(server_proc.stdout):
        raise SshConnectionError()

    client_proc = subprocess.run("ssh {} {}".format(cmdline_args.client_id, cmd),
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT,
               shell=True,
               universal_newlines=True)
    if has_ssh_connection_err_msg(client_proc.stdout):
        raise SshConnectionError()

    return (server_proc.returncode == 0 and client_proc.returncode == 0)

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
    command = "ssh {} 'fi_info -p efa || /opt/amazon/efa/bin/fi_info -p efa'".format(server_id)
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

@functools.lru_cache(10)
@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def get_efa_device_names(server_id):
    timeout = 60

    # This command returns a list of EFA devices names
    command = "ssh {} 'fi_info -p efa -t FI_EP_RDM -f efa || /opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM -f efa' | grep domain".format(server_id)
    proc = subprocess.run(command, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", timeout=timeout)

    if has_ssh_connection_err_msg(proc.stderr):
        raise SshConnectionError()

    devices = []
    stdouts = proc.stdout.strip().split("\n")
    #
    # Example out of fi_info -p efa -t FI_EP_RDM -f efa | grep domain are like the following:
    #     domain: rdmap0s31-rdm
    #     ...
    #
    # Extract the device name from the second column of the stdout
    # by removing the -rdm suffix
    for line in stdouts:
        parts = line.split()
        if len(parts) > 1 and "rdm" in parts[1]:
            devices.append(parts[1].replace("-rdm", ""))
    return devices


def get_efa_device_name_for_cuda_device(ip, cuda_device_id, num_cuda_devices):
    # this function implemented a simple way to find the closest EFA device for a given
    # cuda device. It assumes EFA devices names are in order (which is usually true but not always)
    #
    # For example, one a system with 8 CUDA devies and 4 EFA devices, this function would
    # for GPU 0 and 1, return EFA device 0
    # for GPU 2 and 3, return EFA device 1
    # for GPU 4 and 5, return EFA device 2
    # for GPU 6 and 7, return EFA device 3
    efa_devices = get_efa_device_names(ip)
    num_efa = len(efa_devices)
    return efa_devices[(cuda_device_id * num_efa) // num_cuda_devices]


@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def support_cq_interrupts(hostname):
    """
    determine whether an EFA device supports fi_cq_sread
    This is based on checking if the device has more than 1 completion vector,
    which indicates support for CQ interrupts
    hostname: a host that has efa
    return: True if sread is supported, False otherwise, None if unable to determine
    """
    try:
        efa_devices = get_efa_device_names(hostname)
        if not efa_devices:
            return None
        efa_device_name = efa_devices[0]
    except Exception:
        return None

    command = 'ssh {} ibv_devinfo -d {} -v | grep "num_comp_vectors"'.format(hostname, efa_device_name)
    process = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

    if process.returncode != 0:
        if process.stderr and has_ssh_connection_err_msg(process.stderr):
            print("encountered ssh connection issue")
            raise SshConnectionError()
        return None

    # Example output of ibv_devinfo -d rdmap85s0 -v | grep "num_comp_vectors"
    # num_comp_vectors:		32
    lines = process.stdout.split('\n')
    for line in lines:
        line = line.strip()
        if 'num_comp_vectors' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    comp_vectors = int(parts[1].strip())
                    # Support sread if num_comp_vectors > 1
                    return comp_vectors > 1
                except ValueError:
                    continue

    return None
