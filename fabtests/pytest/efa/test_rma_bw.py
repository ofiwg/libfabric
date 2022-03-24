import pytest

@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_bw(cmdline_args, iteration_type, operation_type, completion_type, memory_type):
    from efa.efa_common import efa_run_client_server_test
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, iteration_type, completion_type, memory_type, "all")

@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
def test_rma_bw_range(cmdline_args, operation_type, completion_type, message_size, memory_type):
    from efa.efa_common import efa_run_client_server_test
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short", completion_type, memory_type, message_size)
