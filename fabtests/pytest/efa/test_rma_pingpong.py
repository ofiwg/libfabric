from efa.efa_common import efa_run_client_server_test
from common import perf_progress_model_cli
import pytest


@pytest.fixture(params=["r:4048,4,4148",
                        "r:8000,4,9000",
                        "r:17000,4,18000"])
def rma_pingpong_message_size(request):
    return request.param


@pytest.mark.parametrize("operation_type", ["writedata"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_pingpong(cmdline_args, iteration_type, operation_type, rma_bw_completion_semantic, memory_type_bi_dir, direct_rma_size, rma_fabric):
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type + " " + perf_progress_model_cli
    efa_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic,
                                memory_type_bi_dir, direct_rma_size if rma_fabric == "efa-direct" else "all", fabric=rma_fabric)


@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata"])
def test_rma_pingpong_range(cmdline_args, operation_type, rma_bw_completion_semantic, rma_pingpong_message_size,
                            direct_rma_size, memory_type_bi_dir, rma_fabric):
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                               memory_type_bi_dir, direct_rma_size if rma_fabric == "efa-direct" else rma_pingpong_message_size, fabric=rma_fabric)


@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata"])
def test_rma_pingpong_range_no_inject(cmdline_args, operation_type, rma_bw_completion_semantic, rma_pingpong_message_size, memory_type_bi_dir, rma_fabric):
    if rma_fabric == "efa-direct":
        pytest.skip("Duplicate test. efa-direct has inject size = 0")
    command = "fi_rma_pingpong -e rdm -j 0"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                                memory_type_bi_dir, rma_pingpong_message_size, fabric=rma_fabric)
