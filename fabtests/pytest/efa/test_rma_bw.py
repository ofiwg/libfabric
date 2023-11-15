from efa.efa_common import efa_run_client_server_test
import pytest


@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_bw(cmdline_args, iteration_type, operation_type, completion_semantic, memory_type):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, iteration_type, completion_semantic, memory_type, "all", timeout=timeout)


@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
def test_rma_bw_range(cmdline_args, operation_type, completion_semantic, message_size, memory_type):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", completion_semantic, memory_type, message_size, timeout=timeout)


@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
def test_rma_bw_range_no_inject(cmdline_args, operation_type, completion_semantic, inject_message_size):
    command = "fi_rma_bw -e rdm -j 0"
    command = command + " -o " + operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", completion_semantic, "host_to_host", inject_message_size, timeout=timeout)


# This test is run in serial mode because it takes a lot of memory
@pytest.mark.serial
@pytest.mark.functional
# TODO Add "writedata", "write" back in when EFA firmware bug is fixed
@pytest.mark.parametrize("operation_type", ["read"])
def test_rma_bw_1G(cmdline_args, operation_type, completion_semantic):
    # Default window size is 64 resulting in 128GB being registered, which
    # exceeds max number of registered host pages
    timeout = max(540, cmdline_args.timeout)
    command = "fi_rma_bw -e rdm -W 1"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, 2,
                               completion_semantic=completion_semantic, message_size=1073741824,
                               memory_type="host_to_host", warmup_iteration_type=0, timeout=timeout)
