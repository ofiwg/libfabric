import pytest
from shm.shm_common import shm_run_client_server_test
from common import (perf_progress_model_cli, num_cuda_devices, PERF_SIZES,
                    PERF_PR_CI, RANGE_SIZES)


@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=PERF_SIZES, pr_ci=PERF_PR_CI)
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_bw(cmdline_args, iteration_type, rma_operation_type, rma_bw_completion_semantic, rma_bw_memory_type, message_sizes):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + rma_operation_type + " " + perf_progress_model_cli
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    shm_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic, rma_bw_memory_type, message_sizes, timeout=timeout)

@pytest.mark.message_sizes(default=RANGE_SIZES)
@pytest.mark.functional
def test_rma_bw_range(cmdline_args, rma_operation_type, rma_bw_completion_semantic, message_sizes, rma_bw_memory_type):
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + rma_operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    shm_run_client_server_test(cmdline_args, command, "short", rma_bw_completion_semantic, rma_bw_memory_type, message_sizes, timeout=timeout)


# shm serves each read between GPUs with a CUDA IPC copy on the target. The
# parameters select that copy path; the data check fails if a read completes
# before its data has landed.
@pytest.mark.functional
@pytest.mark.cuda_memory
@pytest.mark.parametrize("copy_path_env", [
    pytest.param("", id="async_copy"),
    pytest.param("FI_HMEM_CUDA_ENABLE_ASYNC_COPY=0", id="sync_copy"),
    pytest.param("FI_SHM_BUFFER_THRESHOLD=1000000", id="buffer_recv"),
])
def test_rma_bw_read_cuda_ipc(cmdline_args, copy_path_env):
    if num_cuda_devices(cmdline_args.server_id) < 2:
        pytest.skip("needs two GPUs")

    # The buffers skip SYNC_MEMOPS so the copies behave as they do for
    # application allocations, and the server uses the next GPU.
    command = "fi_rma_bw -e rdm -o read -W 1024 --no-cuda-sync-memops"
    # A passing run takes a few seconds. When one side finds bad data it
    # exits and the other side waits for it until the timeout.
    shm_run_client_server_test(cmdline_args, command, "8192",
                               "transmit_complete", "cuda_to_cuda", "16384",
                               warmup_iteration_type="0", timeout=30,
                               additional_env=copy_path_env,
                               hmem_device_offset={"server": 1, "client": 0})
