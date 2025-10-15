from efa.efa_common import efa_run_client_server_test
from common import perf_progress_model_cli
import pytest
import copy


@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_bw(cmdline_args, iteration_type, rma_operation_type, rma_bw_completion_semantic, rma_bw_memory_type, direct_rma_size, rma_fabric, rx_cq_data_cli):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + rma_operation_type + " " + perf_progress_model_cli + rx_cq_data_cli
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic,
                               rma_bw_memory_type, direct_rma_size if rma_fabric == "efa-direct" else "all",
                               timeout=timeout, fabric=rma_fabric)

@pytest.mark.parametrize("env_vars", [["FI_EFA_TX_SIZE=64"], ["FI_EFA_RX_SIZE=64"], ["FI_EFA_TX_SIZE=64", "FI_EFA_RX_SIZE=64"]])
def test_rma_bw_small_tx_rx(cmdline_args, rma_operation_type, rma_bw_completion_semantic, rma_bw_memory_type, env_vars, direct_rma_size, rma_fabric):
    cmdline_args_copy = copy.copy(cmdline_args)
    for env_var in env_vars:
        cmdline_args_copy.append_environ(env_var)
    # Use a window size larger than tx/rx size
    command = "fi_rma_bw -e rdm -W 128"
    command = command + " -o " + rma_operation_type + " " + perf_progress_model_cli
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args_copy.timeout)
    efa_run_client_server_test(cmdline_args_copy, command, "short", rma_bw_completion_semantic,
                               rma_bw_memory_type, direct_rma_size if rma_fabric == "efa-direct" else "all",
                               timeout=timeout, fabric=rma_fabric)

@pytest.mark.functional
def test_rma_bw_range(cmdline_args, rma_operation_type, rma_bw_completion_semantic, message_size, direct_rma_size, rma_bw_memory_type, rma_fabric):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + rma_operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(1080, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                               rma_bw_memory_type, direct_rma_size if rma_fabric == "efa-direct" else message_size,
                               timeout=timeout, fabric=rma_fabric)


@pytest.mark.functional
def test_rma_bw_range_no_inject(cmdline_args, rma_operation_type, rma_bw_completion_semantic, inject_message_size, rma_fabric):
    if rma_fabric == "efa-direct":
        pytest.skip("Duplicate test. efa-direct has inject size = 0")
    command = "fi_rma_bw -e rdm -j 0"
    command = command + " -o " + rma_operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                                "host_to_host", inject_message_size, timeout=timeout, fabric=rma_fabric)


# This test is run in serial mode because it takes a lot of memory
@pytest.mark.serial
@pytest.mark.functional
# TODO Add "writedata", "write" back in when EFA firmware bug is fixed
# TODO enable efa-direct test after fixing fabtests to post recv within device max msg size.
@pytest.mark.parametrize("operation_type", ["read"])
def test_rma_bw_1G(cmdline_args, operation_type, rma_bw_completion_semantic):
    # Default window size is 64 resulting in 128GB being registered, which
    # exceeds max number of registered host pages
    timeout = max(540, cmdline_args.timeout)
    command = "fi_rma_bw -e rdm -W 1"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, 2,
                               completion_semantic=rma_bw_completion_semantic, message_size=1073741824,
                               memory_type="host_to_host", warmup_iteration_type=0, timeout=timeout, fabric="efa")

@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata", "write"])
@pytest.mark.parametrize("iteration_type",
                         ["5", # smaller than max batch wqe cnt (16)
                          "48", # larger than max batch wqe cnt
                          "128"]) # larger than window size (64)
def test_rma_bw_use_fi_more(cmdline_args, operation_type, iteration_type, rma_bw_completion_semantic, inject_message_size, direct_rma_size, rma_fabric):
    command = "fi_rma_bw -e rdm -j 0 --use-fi-more"
    command = command + " -o " + operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic,
                               "host_to_host", direct_rma_size if rma_fabric == "efa-direct" else inject_message_size,
                               timeout=timeout, fabric=rma_fabric)


@pytest.mark.functional
@pytest.mark.parametrize("comp_method", ["sread", "fd"])
def test_rma_bw_sread(cmdline_args, rma_operation_type, rma_bw_completion_semantic,
                      direct_rma_size, rma_bw_memory_type, support_sread, comp_method, rma_fabric):
    if not support_sread:
        pytest.skip("sread not supported by efa device.")
    if rma_fabric == "efa":
        pytest.skip("sread not implemented in efa fabric yet.")
    command = f"fi_rma_bw -e rdm -c {comp_method}"
    command = command + " -o " + rma_operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(1080, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                               rma_bw_memory_type, direct_rma_size,
                               timeout=timeout, fabric=rma_fabric)
