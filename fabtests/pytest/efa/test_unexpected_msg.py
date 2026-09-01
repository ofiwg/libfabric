import pytest
from efa.efa_common import efa_run_client_server_test, memory_type_list_all, memory_type_list_device_to_device


SHM_DEFAULT_MAX_INJECT_SIZE = 4096
SHM_DEFAULT_RX_SIZE = 1024


# This test skips efa-direct because it does not have unexpected message
@pytest.mark.pr_ci
@pytest.mark.pr_ci_hmem
@pytest.mark.functional
@pytest.mark.parametrize("msg_size", [1, 512, 9000, 1048576]) # cover various switch points of shm/efa protocols
@pytest.mark.parametrize("msg_count", [1, 1024, 2048]) # below and above shm's default rx size
@pytest.mark.memory_type(memory_type_list_all, pr_ci_hmem=memory_type_list_device_to_device)
def test_unexpected_msg(cmdline_args, msg_size, msg_count, memory_type, completion_semantic, request):
    from common import ClientServerTest, test_selected_by_marker
    if cmdline_args.server_id == cmdline_args.client_id:
        if (msg_size > SHM_DEFAULT_MAX_INJECT_SIZE or memory_type != "host_to_host" or completion_semantic == "delivery_complete") and msg_count > SHM_DEFAULT_RX_SIZE:
            pytest.skip("SHM's CMA/IPC protocol currently cannot handle > rx size number of unexpected messages")
    # This fabtests will allocate msg_size * 2 * msg_count memory for send/recv
    allocated_memory = msg_size * 2 * msg_count
    # The limit size (4 GB) of neuron_tensor_alloc
    neuron_maximal_buffer_size = 2**32
    if "neuron" in memory_type and allocated_memory >= neuron_maximal_buffer_size:
        pytest.skip("Cannot hit neuron allocation limit")

     # The EFA limit for single MR that enables remote write is 1M pages aka 4GB for regular pages
    maximal_mr_size = 2**32
    if allocated_memory >= maximal_mr_size:
        pytest.skip("Cannot hit EFA MR limit")

    # A 2GB working set (1MB x 1024) costs 0.8 to 1.6 min per case on device
    # memory, because every transfer bounces through a host buffer on instances
    # without GPUDirect RDMA. The protocol under test does not change with the
    # working set size, so skip it in PR CI. host_to_host still covers every
    # size and the nightly runs keep the full matrix.
    pr_ci_maximal_hmem_size = 2**30
    is_pr_ci = test_selected_by_marker(request.config,
                                       {m.name for m in request.node.iter_markers()},
                                       "pr_ci")
    if is_pr_ci and memory_type != "host_to_host" and allocated_memory >= pr_ci_maximal_hmem_size:
        pytest.skip("Device memory working set is too slow for PR CI")

    efa_run_client_server_test(cmdline_args, f"fi_unexpected_msg -e rdm -M {msg_count}", iteration_type="short",
                               completion_semantic=completion_semantic, memory_type=memory_type,
                               message_size=msg_size, completion_type="queue", timeout=1800, fabric="efa")
