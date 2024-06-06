from efa.efa_common import efa_run_client_server_test
from common import perf_progress_model_cli
import pytest


@pytest.mark.parametrize("operation_type", ["writedata", "write"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_pingpong(cmdline_args, iteration_type, operation_type, completion_semantic, memory_type):
    if memory_type != "host_to_host" and operation_type == "write":
        pytest.skip("no hmem memory support for pingpong_rma write test")
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type + " " + perf_progress_model_cli
    # rma_pingpong test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, iteration_type, completion_semantic, memory_type, "all", timeout=timeout)


@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata", "write"])
def test_rma_pingpong_range(cmdline_args, operation_type, completion_semantic, message_size, memory_type):
    if memory_type != "host_to_host" and operation_type == "write":
        pytest.skip("no hmem memory support for pingpong_rma write test")
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type
    # rma_pingpong test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", completion_semantic, memory_type, message_size, timeout=timeout)


@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata", "write"])
def test_rma_pingpong_range_no_inject(cmdline_args, operation_type, completion_semantic, inject_message_size, memory_type):
    if memory_type != "host_to_host" and operation_type == "write":
        pytest.skip("no hmem memory support for pingpong_rma write test")
    command = "fi_rma_pingpong -e rdm -j 0"
    command = command + " -o " + operation_type
    # rma_pingpong test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, "short", completion_semantic, "host_to_host", inject_message_size, timeout=timeout)
