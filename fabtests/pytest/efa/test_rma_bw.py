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
