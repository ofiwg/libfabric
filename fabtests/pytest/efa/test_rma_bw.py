from efa.efa_common import (efa_run_client_server_test, DIRECT_SIZES,
                            memory_type_list_all, memory_type_list_device_to_device,
                            CudaMemorySupport, get_cuda_memory_support,
                            get_efa_devices_on_dma_path)
from common import (perf_progress_model_cli, ClientServerTest,
                    PERF_SIZES, PERF_PR_CI, RANGE_SIZES, INJECT_SIZES,
                    NIC_DMA_PATH_CPU_MEDIATED)
import pytest
import copy


@pytest.mark.pr_ci
@pytest.mark.pr_ci_hmem
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=PERF_SIZES, default_efa_direct=DIRECT_SIZES,
                           pr_ci_efa=PERF_PR_CI, pr_ci_efa_direct=DIRECT_SIZES)
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
@pytest.mark.memory_type(memory_type_list_all, pr_ci_hmem=memory_type_list_device_to_device)
def test_rma_bw(cmdline_args, iteration_type, rma_operation_type, rma_bw_completion_semantic, rma_bw_memory_type, rma_fabric, rx_cq_data_cli, message_sizes):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + rma_operation_type + " " + perf_progress_model_cli + rx_cq_data_cli
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic,
                               rma_bw_memory_type, message_sizes,
                               timeout=timeout, fabric=rma_fabric)

@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=PERF_SIZES, default_efa_direct=DIRECT_SIZES)
@pytest.mark.parametrize("env_vars", [["FI_EFA_TX_SIZE=64"], ["FI_EFA_RX_SIZE=64"], ["FI_EFA_TX_SIZE=64", "FI_EFA_RX_SIZE=64"]])
@pytest.mark.memory_type(memory_type_list_all)
def test_rma_bw_small_tx_rx(cmdline_args, rma_operation_type, rma_bw_completion_semantic, rma_bw_memory_type, env_vars, rma_fabric, message_sizes):
    cmdline_args_copy = copy.copy(cmdline_args)
    for env_var in env_vars:
        cmdline_args_copy.append_environ(env_var)
    # Use a window size larger than tx/rx size
    command = "fi_rma_bw -e rdm -W 128"
    command = command + " -o " + rma_operation_type + " " + perf_progress_model_cli
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args_copy.timeout)
    efa_run_client_server_test(cmdline_args_copy, command, "short", rma_bw_completion_semantic,
                               rma_bw_memory_type, message_sizes,
                               timeout=timeout, fabric=rma_fabric)

@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=RANGE_SIZES, default_efa_direct=DIRECT_SIZES)
@pytest.mark.functional
@pytest.mark.memory_type(memory_type_list_all)
def test_rma_bw_range(cmdline_args, rma_operation_type, rma_bw_completion_semantic, message_sizes, rma_bw_memory_type, rma_fabric, completion_type):
    if completion_type == "counter" and rma_operation_type == "writedata":
        pytest.skip("writedata target cannot track remote-write completions with counters")
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + rma_operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(1080, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                               rma_bw_memory_type, message_sizes, completion_type=completion_type,
                               timeout=timeout, fabric=rma_fabric)


@pytest.mark.fabric(params=["efa"])
@pytest.mark.message_sizes(default_efa=INJECT_SIZES)
@pytest.mark.functional
def test_rma_bw_range_no_inject(cmdline_args, rma_operation_type, rma_bw_completion_semantic, message_sizes, rma_fabric):
    command = "fi_rma_bw -e rdm -j 0"
    command = command + " -o " + rma_operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                                "host_to_host", message_sizes, timeout=timeout, fabric=rma_fabric)


# This test is run in serial mode because it takes a lot of memory
@pytest.mark.serial
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["read", "write", "writedata"])
def test_rma_bw_1G(cmdline_args, operation_type, rma_bw_completion_semantic, rma_fabric):
    # Default window size is 64 resulting in 128GB being registered, which
    # exceeds max number of registered host pages.
    # Use a single iteration without warmup or data verification to keep
    # CI time reasonable. Data integrity for large RMA operations is
    # covered by test_rma_bw_large below.
    timeout = max(540, cmdline_args.timeout)
    command = "fi_rma_bw -e rdm -W 1 -I 1 -w 0"
    command = command + " -o " + operation_type
    test = ClientServerTest(cmdline_args, command, iteration_type=None,
                            completion_semantic=rma_bw_completion_semantic,
                            datacheck_type="wout_datacheck",
                            message_size=1073741824,
                            memory_type="host_to_host",
                            timeout=timeout,
                            fabric=rma_fabric)
    test.run()


@pytest.mark.serial
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["read", "write", "writedata"])
def test_rma_bw_large(cmdline_args, operation_type, rma_bw_completion_semantic, rma_fabric):
    # Verify data integrity for large RMA operations using 64MB messages.
    # This covers the same large-message code paths as 1G but completes
    # fast enough for ASAN builds.
    timeout = max(540, cmdline_args.timeout)
    command = "fi_rma_bw -e rdm -W 1"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, 2,
                               completion_semantic=rma_bw_completion_semantic, message_size=67108864,
                               memory_type="host_to_host", warmup_iteration_type=0, timeout=timeout, fabric=rma_fabric)

@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=INJECT_SIZES, default_efa_direct=DIRECT_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata", "write"])
@pytest.mark.parametrize("iteration_type",
                         ["5", # smaller than max batch wqe cnt (16)
                          "48", # larger than max batch wqe cnt
                          "128"]) # larger than window size (64)
def test_rma_bw_use_fi_more(cmdline_args, operation_type, iteration_type, rma_bw_completion_semantic, message_sizes, rma_fabric):
    command = "fi_rma_bw -e rdm -w 0 -j 0 --use-fi-more --sync-comp " + rma_bw_completion_semantic
    command = command + " -o " + operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic,
                               "host_to_host", message_sizes,
                               timeout=timeout, fabric=rma_fabric)


@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=PERF_SIZES, default_efa_direct=DIRECT_SIZES,
                           pr_ci_efa=PERF_PR_CI, pr_ci_efa_direct=DIRECT_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("comp_method", ["sread", "fd"])
@pytest.mark.memory_type(memory_type_list_all)
def test_rma_bw_sread(cmdline_args, rma_operation_type, rma_bw_completion_semantic,
                      rma_bw_memory_type, support_sread, comp_method,
                      rma_fabric, message_sizes):
    if not support_sread:
        pytest.skip("sread not supported by efa device.")
    additional_env = ''
    if rma_fabric == "efa" and comp_method == "fd":
        if cmdline_args.server_id == cmdline_args.client_id:
            pytest.skip("FI_WAIT_FD not supported for EFA protocol with SHM enabled")
        additional_env = "FI_EFA_ENABLE_SHM_TRANSFER=0"
    command = f"fi_rma_bw -e rdm -c {comp_method}"
    command = command + " -o " + rma_operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(1080, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                               rma_bw_memory_type, message_sizes,
                               timeout=timeout, fabric=rma_fabric, additional_env=additional_env)


@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=PERF_SIZES, default_efa_direct=DIRECT_SIZES,
                           pr_ci_efa=PERF_PR_CI, pr_ci_efa_direct=DIRECT_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["write", "writedata"])
# Only test host and cuda memory; other HMEM types do not change the RMA path.
@pytest.mark.parametrize("mem_type",
                         ["host_to_host",
                          pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory)])
def test_efa_rma_bw_high_pps(cmdline_args, operation_type, mem_type, rma_fabric):
    command = "fi_efa_rma_bw -e rdm --high-pps"
    command += " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short",
                               completion_semantic="transmit_complete",
                               memory_type=mem_type,
                               message_size="all",
                               fabric=rma_fabric,
                               additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")


# Testing the batch mode of fi_efa_rma_bw (--post-list) which batch multiple WQEs with FI_MORE
@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=PERF_SIZES, default_efa_direct=DIRECT_SIZES,
                           pr_ci_efa=PERF_PR_CI, pr_ci_efa_direct=DIRECT_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["write", "writedata"])
# Only test host and cuda memory; other HMEM types do not change the RMA path.
@pytest.mark.parametrize("mem_type",
                         ["host_to_host",
                          pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory)])
def test_efa_rma_bw_batch(cmdline_args, operation_type, mem_type, rma_fabric):
    command = "fi_efa_rma_bw -e rdm --post-list 16"
    command += " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short",
                               completion_semantic="transmit_complete",
                               memory_type=mem_type,
                               message_size="all",
                               fabric=rma_fabric,
                               additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")


# Testing fi_efa_rma_bw with the EFA-specific FI_EFA_MR_RELAXED_ORDERING MR flag
# (--mr-relaxed-ordering). The flag is honored on efa-direct and silently
# ignored on efa, so registration and the transfer must succeed on both fabrics.
@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=PERF_SIZES, default_efa_direct=DIRECT_SIZES,
                           pr_ci_efa=PERF_PR_CI, pr_ci_efa_direct=DIRECT_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["read", "write", "writedata"])
# Only test host and cuda memory; other HMEM types do not change the RMA path.
@pytest.mark.parametrize("mem_type",
                         ["host_to_host",
                          pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory)])
def test_efa_rma_bw_mr_relaxed_ordering(cmdline_args, operation_type, mem_type, rma_fabric):
    command = "fi_efa_rma_bw -e rdm --mr-relaxed-ordering"
    command += " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short",
                               completion_semantic="transmit_complete",
                               memory_type=mem_type,
                               message_size="all",
                               fabric=rma_fabric,
                               additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")


# A CUDA dmabuf fd encodes one of the two DMA paths to GPU memory, and the two
# need different mapping types: a NIC that shares a PCIe switch with the GPU
# needs CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE, while a NIC that reaches HBM
# across the CPU needs the default mapping. Asking for the wrong one is not an
# error at registration time, it just yields a handle to the wrong physical
# address, so the mistake only surfaces as unresponsive-remote errors once data
# moves.
#
# Every other CUDA test runs over a GPU-local NIC, because that is the NIC the
# PCIe traversal in get_efa_device_name_for_cuda_device() finds, so nothing else
# exercises the other mapping. Only a platform that has both kinds of NIC has
# anything to run here: p6e-gb200, whose second set of NICs reaches GPU memory
# over NVLink-C2C, is the one today, and this skips everywhere else.
@pytest.mark.functional
@pytest.mark.cuda_memory
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.parametrize("operation_type", ["read", "write"])
def test_rma_bw_cuda_dmabuf_over_cpu_mediated_nic(cmdline_args, operation_type, rma_fabric):
    # The mapping type is only chosen where fabtests exports the dmabuf fd
    # itself, which it does under -R, so without it there is nothing to cover.
    if not cmdline_args.do_dmabuf_reg_for_hmem:
        pytest.skip("this test needs dmabuf registration to be enabled")

    for host in (cmdline_args.server_id, cmdline_args.client_id):
        if not get_efa_devices_on_dma_path(host, NIC_DMA_PATH_CPU_MEDIATED):
            pytest.skip("{} has no NIC that reaches GPU memory across the CPU".format(host))
        if get_cuda_memory_support(cmdline_args, host) not in (CudaMemorySupport.DMA_BUF_ONLY,
                                                              CudaMemorySupport.DMABUF_GDR_BOTH):
            pytest.skip("{} does not support CUDA dmabuf".format(host))

    command = "fi_rma_bw -e rdm -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short",
                               completion_semantic="transmit_complete",
                               memory_type="cuda_to_cuda",
                               message_size="all",
                               fabric=rma_fabric,
                               nic_dma_path=NIC_DMA_PATH_CPU_MEDIATED,
                               timeout=max(540, cmdline_args.timeout))
