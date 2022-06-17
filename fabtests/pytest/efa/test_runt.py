import pytest

@pytest.mark.functional
def test_runt_read_functional(cmdline_args):
    """
    Verify runt reading protocol is working as expected by sending 1 message of 256 KB.
    64 KB of the message will be transfered using EFA device's send capability
    The remainer will be tranfered using EFA device's RDMA read capability
    """

    import copy
    from efa.efa_common import efa_run_client_server_test, efa_retrieve_hw_counter_value, has_gdrcopy

    cmdline_args_copy = copy.copy(cmdline_args)

    if cmdline_args_copy.environments:
        cmdline_args_copy.environments += " "
    else:
        cmdline_args_copy.environments = ""

    cmdline_args_copy.environments += "FI_EFA_USE_DEVICE_RDMA_READ=1 FI_EFA_RUNT_SIZE=65536 FI_HMEM_CUDA_USE_GDRCOPY=1"

    # currently, runting read is enabled only if gdrcopy is available.
    # thus skip the test if gdrcopy is not available
    if not has_gdrcopy(cmdline_args.server_id) or not has_gdrcopy(cmdline_args.client_id):
        pytest.skip("No gdrcopy")
        return
 
    # wrs stands for work requests
    server_read_wrs_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_wrs")
    server_read_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_bytes")
    client_send_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "send_bytes")

    # currently runting read is only used on cuda memory, hence set the memory_type to "cuda_to_cuda"
    efa_run_client_server_test(cmdline_args_copy,
                               "fi_rdm_tagged_bw",
                               iteration_type="1",
                               completion_type="transmit_complete",
                               memory_type="cuda_to_cuda",
                               message_size="262144",
                               warmup_iteration_type="0")

    server_read_wrs_after_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_wrs")
    server_read_bytes_after_test =efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_bytes")
    client_send_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "send_bytes")

    print("server_read_bytes_before_test: {}".format(server_read_bytes_before_test))
    print("server_read_bytes_after_test: {}".format(server_read_bytes_after_test))

    if server_read_bytes_after_test == 0:
        # Not all instance types support RDMA read.
        # return here for those that does not support
        return

    # fi_rdm_tagged_bw is a uni-diretional test. client is the sender and server is the receiver.
    # In a RDMA read based message tranfer protocol, the receiver will submit work request for read
    server_read_wrs = server_read_wrs_after_test - server_read_wrs_before_test
    server_read_bytes = server_read_bytes_after_test - server_read_bytes_before_test
    client_send_bytes = client_send_bytes_after_test - client_send_bytes_before_test

    # Among the 256 KB of data, 64 KB data will be sent via 8 RUNTREAD RTM packets.
    # Each packet has a packet header, therefore the total number of send bytes
    # is slightly larger than 64 K.
    assert client_send_bytes > 65536 and client_send_bytes < 66560

    # The other 192 KB is transfer by RDMA read
    # for which the server (receiver) will issue 1 read request.
    assert server_read_wrs == 1
    assert server_read_bytes == 196608
