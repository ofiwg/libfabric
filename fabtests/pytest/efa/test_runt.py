import pytest

# this test must be run in serial mode because it check hw counter
@pytest.mark.serial
@pytest.mark.functional
@pytest.mark.parametrize("cuda_copy_method", ["gdrcopy", "localread"])
def test_runt_read_functional(cmdline_args, cuda_copy_method):
    """
    Verify runt reading protocol is working as expected by sending 1 message of 256 KB.
    64 KB of the message will be transfered using EFA device's send capability
    The remainer will be tranfered using EFA device's RDMA read capability
    """

    import copy
    from efa.efa_common import efa_run_client_server_test, efa_retrieve_hw_counter_value, has_gdrcopy

    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("no runting for intra-node communication")

    cmdline_args_copy = copy.copy(cmdline_args)

    cmdline_args_copy.append_environ("FI_EFA_USE_DEVICE_RDMA=1")
    cmdline_args_copy.append_environ("FI_EFA_RUNT_SIZE=65536")

    if cuda_copy_method == "gdrcopy":
        if not has_gdrcopy(cmdline_args.server_id) or not has_gdrcopy(cmdline_args.client_id):
            pytest.skip("No gdrcopy")
            return

        cmdline_args_copy.append_environ("FI_HMEM_CUDA_USE_GDRCOPY=1")
    else:
        cmdline_args_copy.append_environ("FI_HMEM_CUDA_USE_GDRCOPY=0")

    # wrs stands for work requests
    server_read_wrs_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_wrs")
    if server_read_wrs_before_test is None:
        pytest.skip("No HW counter support")
        return

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
    # The total number of send bytes will be larger than 64K because:
    #    a. each packet has a header
    #    b. when runing on single node, server will use the same EFA device to send control packets
    assert client_send_bytes > 65536

    if cuda_copy_method == "gdrcopy":
        # The other 192 KB is transfer by RDMA read
        # for which the server (receiver) will issue 1 read request.
        assert server_read_wrs == 1
        assert server_read_bytes == 196608
    else:
        # when local read copy is used, server issue RDMA requests to copy received data
        #
        # so in this case, total read wr is 11, which is
        #    1 remote read of 192k
        #    8 local read for the 64k data transfer by send
        #    2 local read for 2 fabtests control messages
        #
        # and total read_bytes will be 262149, which is:
        #        256k message + 2 fabtests control messages (1 byte and 4 byte each)
        #
        assert server_read_wrs == 11
        assert server_read_bytes == 262149
