import pytest
from efa.efa_common import efa_run_client_server_test

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
@pytest.mark.functional
def test_unexpected_msg(cmdline_args, iteration_type, completion_semantic, memory_type, message_size):
    if completion_semantic == "delivery_complete":
        pytest.skip("Unexpected message test does not support delivery_complete completion semantic")

    # SHM provider currently supports 1 in flight message for host<->device transfers
    if memory_type in ["cuda_to_host", "host_to_cuda"]:
        inflight_msgs = 1
    else:
        inflight_msgs = 4

    efa_run_client_server_test(cmdline_args, f"fi_unexpected_msg -M {inflight_msgs}", iteration_type,
                               completion_semantic, memory_type, message_size)
