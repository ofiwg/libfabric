from efa.efa_common import efa_run_client_server_test, trim_to_efa_direct_message_sizes
from common import perf_progress_model_cli, PERF_SIZES, PERF_PR_CI, RMA_PINGPONG_SIZES
import pytest


@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=PERF_SIZES, pr_ci=PERF_PR_CI)
@pytest.mark.parametrize("operation_type", ["writedata"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_pingpong(cmdline_args, iteration_type, operation_type, rma_bw_completion_semantic, memory_type_bi_dir, rma_fabric, message_sizes):
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type + " " + perf_progress_model_cli
    if rma_fabric == "efa-direct":
        message_sizes = trim_to_efa_direct_message_sizes(message_sizes)
    efa_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic,
                                memory_type_bi_dir, message_sizes, fabric=rma_fabric)


@pytest.mark.message_sizes(default=RMA_PINGPONG_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata"])
def test_rma_pingpong_range(cmdline_args, operation_type, rma_bw_completion_semantic, message_sizes,
                            memory_type_bi_dir, rma_fabric):
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type
    if rma_fabric == "efa-direct":
        message_sizes = trim_to_efa_direct_message_sizes(message_sizes)
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                               memory_type_bi_dir, message_sizes, fabric=rma_fabric)


@pytest.mark.message_sizes(default=RMA_PINGPONG_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata"])
def test_rma_pingpong_range_no_inject(cmdline_args, operation_type, rma_bw_completion_semantic, message_sizes, memory_type_bi_dir, rma_fabric):
    if rma_fabric == "efa-direct":
        pytest.skip("Duplicate test. efa-direct has inject size = 0")
    command = "fi_rma_pingpong -e rdm -j 0"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                                memory_type_bi_dir, message_sizes, fabric=rma_fabric)
