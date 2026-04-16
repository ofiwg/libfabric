import pytest
from shm.shm_common import shm_run_client_server_test
from common import perf_progress_model_cli, PERF_SIZES, PERF_PR_CI, RANGE_SIZES


@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=PERF_SIZES, pr_ci=PERF_PR_CI)
@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_bw(cmdline_args, iteration_type, operation_type, completion_semantic, memory_type, message_sizes):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + operation_type + " " + perf_progress_model_cli
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    shm_run_client_server_test(cmdline_args, command, iteration_type, completion_semantic, memory_type, message_sizes, timeout=timeout)

@pytest.mark.message_sizes(default=RANGE_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
def test_rma_bw_range(cmdline_args, operation_type, completion_semantic, message_sizes, memory_type):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    shm_run_client_server_test(cmdline_args, command, "short", completion_semantic, memory_type, message_sizes, timeout=timeout)
