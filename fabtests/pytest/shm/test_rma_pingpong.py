import pytest
from shm.shm_common import shm_run_client_server_test
from common import perf_progress_model_cli, PERF_SIZES, PERF_PR_CI, RMA_PINGPONG_SIZES


@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=PERF_SIZES, pr_ci=PERF_PR_CI)
@pytest.mark.parametrize("operation_type", ["writedata"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_pingpong(cmdline_args, iteration_type, operation_type, completion_semantic, memory_type, message_sizes):
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type + " " + perf_progress_model_cli
    shm_run_client_server_test(cmdline_args, command, iteration_type, completion_semantic, memory_type, message_sizes)

@pytest.mark.message_sizes(default=RMA_PINGPONG_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata"])
def test_rma_pingpong_range(cmdline_args, operation_type, completion_semantic, message_sizes, memory_type):
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type
    shm_run_client_server_test(cmdline_args, command, "short", completion_semantic, memory_type, message_sizes)
