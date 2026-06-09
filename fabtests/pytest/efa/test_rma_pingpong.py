from efa.efa_common import efa_run_client_server_test, DIRECT_RMA_SIZES
from common import perf_progress_model_cli, PERF_SIZES, PERF_PR_CI, RMA_PINGPONG_SIZES
import pytest


@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=PERF_SIZES, default_efa_direct=DIRECT_RMA_SIZES,
                           pr_ci_efa=PERF_PR_CI, pr_ci_efa_direct=DIRECT_RMA_SIZES)
@pytest.mark.parametrize("operation_type", ["writedata", "read"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_pingpong(cmdline_args, iteration_type, operation_type, rma_bw_completion_semantic, memory_type_bi_dir, rma_fabric, message_sizes):
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type + " " + perf_progress_model_cli
    efa_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic,
                                memory_type_bi_dir, message_sizes, fabric=rma_fabric)


@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=RMA_PINGPONG_SIZES, default_efa_direct=DIRECT_RMA_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata"])
def test_rma_pingpong_range(cmdline_args, operation_type, rma_bw_completion_semantic, message_sizes,
                            memory_type_bi_dir, rma_fabric):
    command = "fi_rma_pingpong -e rdm"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                               memory_type_bi_dir, message_sizes, fabric=rma_fabric)


@pytest.mark.fabric(params=["efa"])
@pytest.mark.message_sizes(default_efa=RMA_PINGPONG_SIZES)
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata"])
def test_rma_pingpong_range_no_inject(cmdline_args, operation_type, rma_bw_completion_semantic, message_sizes, memory_type_bi_dir, rma_fabric):
    command = "fi_rma_pingpong -e rdm -j 0"
    command = command + " -o " + operation_type
    efa_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic,
                                memory_type_bi_dir, message_sizes, fabric=rma_fabric)


@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata"])
@pytest.mark.parametrize("inject_size", [64])
@pytest.mark.parametrize("use_data_path_direct", [0, 1])
def test_rma_pingpong_wide_wqe(cmdline_args, operation_type, inject_size, rma_fabric, use_data_path_direct):
    """Test RMA pingpong with wide WQE inject."""
    command = "fi_rma_pingpong -e rdm -E -o {} -j {} -S {}".format(operation_type, inject_size, inject_size)
    env = "FI_EFA_USE_DATA_PATH_DIRECT={}".format(use_data_path_direct)
    efa_run_client_server_test(cmdline_args, command, "short", "delivery_complete","host_to_host", "all", fabric=rma_fabric, additional_env=env)
