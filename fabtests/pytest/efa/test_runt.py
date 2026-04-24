from efa.efa_common import (efa_retrieve_hw_counter_value,
                            efa_run_client_server_test, has_gdrcopy)

import pytest


# this test must be run in serial mode because it check hw counter
# efa-direct does not have runt read so skip this test
@pytest.mark.serial
@pytest.mark.functional
@pytest.mark.parametrize("memory_type,copy_method", [
    pytest.param("cuda_to_cuda", "gdrcopy", marks=pytest.mark.cuda_memory),
    pytest.param("cuda_to_cuda", "localread", marks=pytest.mark.cuda_memory),
    pytest.param("neuron_to_neuron", None, marks=pytest.mark.neuron_memory)])
def test_runt_read_functional(cmdline_args, memory_type, copy_method):
    """
    Verify runt read protocol works with FI_OPT_EFA_HOMOGENEOUS_PEERS set,
    which skips the handshake. This sends 1 message of 256 KB using
    fi_efa_runt_read_no_handshake (a unidirectional bandwidth test).

    64 KB of the message is transferred using EFA device's send capability (runt).
    The remainder (192 KB) is transferred using EFA device's RDMA read capability.

    The test uses FI_OPT_EFA_HOMOGENEOUS_PEERS to skip the handshake, verifying
    that the runt read protocol can be selected without prior handshake exchange.
    """
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("no runting for intra-node communication")

    additional_env = "FI_EFA_RUNT_SIZE=65536 FI_EFA_ENABLE_SHM_TRANSFER=0"

    if memory_type == "host_to_host":
        # For host memory, min_read_message_size defaults to 1MB.
        # Lower it so the 256KB message triggers the read-based path.
        additional_env += " FI_EFA_INTER_MIN_READ_MESSAGE_SIZE=65536"

    if copy_method == "gdrcopy":
        if not has_gdrcopy(cmdline_args.server_id) or not has_gdrcopy(cmdline_args.client_id):
            pytest.skip("No gdrcopy")
        additional_env += " FI_HMEM_CUDA_USE_GDRCOPY=1"
    elif copy_method == "localread":
        assert memory_type == "cuda_to_cuda"
        additional_env += " FI_HMEM_CUDA_USE_GDRCOPY=0"

    # wrs stands for work requests
    server_read_wrs_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_wrs")
    if server_read_wrs_before_test is None:
        pytest.skip("No HW counter support")

    server_read_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_bytes")
    client_send_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "send_bytes")

    efa_run_client_server_test(cmdline_args,
                               "fi_efa_runt_read_no_handshake",
                               iteration_type="1",
                               completion_semantic="transmit_complete",
                               memory_type=memory_type,
                               message_size="262144",
                               warmup_iteration_type="0",
                               fabric="efa",
                               additional_env=additional_env)

    server_read_wrs_after_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_wrs")
    server_read_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_bytes")
    client_send_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "send_bytes")

    print("server_read_bytes_before_test: {}".format(server_read_bytes_before_test))
    print("server_read_bytes_after_test: {}".format(server_read_bytes_after_test))

    if server_read_bytes_after_test == 0:
        # Not all instance types support RDMA read.
        # return here for those that does not support
        return

    # fi_efa_runt_read_no_handshake is a unidirectional test.
    # Client is the sender and server is the receiver.
    # In the runt read protocol, the receiver issues RDMA read work requests.
    server_read_wrs = server_read_wrs_after_test - server_read_wrs_before_test
    server_read_bytes = server_read_bytes_after_test - server_read_bytes_before_test
    client_send_bytes = client_send_bytes_after_test - client_send_bytes_before_test

    # Among the 256 KB of data, 64 KB data will be sent via RUNTREAD RTM packets.
    # The total number of send bytes will be larger than 64K because:
    #    a. each packet has a header
    #    b. when running on single node, server will use the same EFA device to send control packets
    assert client_send_bytes > 65536

    if copy_method == "localread":
        # when local read copy is used, server issues RDMA requests to copy received data
        #
        # so in this case, total read wr is at least 9, which is
        #    1 remote read of 192k
        #    8 local read for the 64k data transferred by send
        #    More local reads for fabtests control messages
        #
        # and total read_bytes will be >= 256K including the control messages
        assert server_read_wrs >= 9
        assert server_read_bytes >= 262144
    else:
        # The other 192 KB is transferred by RDMA read
        # for which the server (receiver) will issue 1 read request.
        assert server_read_wrs == 1
        assert server_read_bytes == 196608
