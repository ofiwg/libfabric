import pytest

from efa.efa_common import has_rdma


# This test skips efa-direct because it does not have the read protocol
# TODO Expand this test to run on all memory types (and rename)
@pytest.mark.serial
@pytest.mark.functional
@pytest.mark.cuda_memory
@pytest.mark.parametrize("fabtest_name", ["fi_efa_runt_read_no_handshake", "fi_rma_bw"])
def test_transfer_with_read_protocol_cuda(cmdline_args, fabtest_name):
    """
    Verify read protocol selection through the send force option or RMA bounds.
    """
    import copy
    from common import has_cuda, has_hmem_support
    from efa.efa_common import efa_run_client_server_test, efa_retrieve_hw_counter_value, has_cuda_p2p

    if not has_rdma(cmdline_args, "read"):
        pytest.skip("RDMA read not supported, cannot test read protocol")

    if fabtest_name == "fi_rma_bw" and has_rdma(cmdline_args, "write"):
        pytest.skip("longread_write is only used for emulated writes")

    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("No read for intra-node communication")

    if (not has_hmem_support(cmdline_args, cmdline_args.client_id) or
        not has_hmem_support(cmdline_args, cmdline_args.server_id)):
        pytest.skip("Client and server both need hmem support for RDMA Read test")

    if not has_cuda(cmdline_args.client_id) or not has_cuda(cmdline_args.server_id):
        pytest.skip("Client and server both need a Cuda device")

    if not has_cuda_p2p(cmdline_args.client_id, cmdline_args.environments) or not has_cuda_p2p(cmdline_args.server_id, cmdline_args.environments):
        pytest.skip("Client and server both need CUDA P2P")

    message_size = 1024

    cmdline_args_copy = copy.copy(cmdline_args)
    cmdline_args_copy.append_environ("FI_EFA_USE_DEVICE_RDMA=1")
    if fabtest_name == "fi_rma_bw":
        cmdline_args_copy.append_environ(f"FI_EFA_INTER_MAX_EAGER_WRITE_SIZE={message_size - 1}")
        cmdline_args_copy.append_environ("FI_EFA_INTER_MIN_READ_WRITE_SIZE=1000")
    else:
        cmdline_args_copy.append_environ("FI_EFA_RDM_FORCE_SEND_PROTO=longread")

    # Use out-of-band socket synchronization (-b) so ft_finalize_ep does not
    # send a libfabric control message during the measured interval, isolating
    # the payload's contribution to rdma_read_bytes.
    cmdline_args_copy.additional_server_arguments = "-b"
    cmdline_args_copy.additional_client_arguments = "-b"

    # wrs stands for work requests
    server_read_wrs_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_wrs")
    if server_read_wrs_before_test is None:
        pytest.skip("No HW counter support")
        return
    server_read_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_bytes")

    efa_run_client_server_test(cmdline_args_copy,
                               fabtest_name,
                               iteration_type="1",
                               completion_semantic="transmit_complete",
                               memory_type="cuda_to_cuda",
                               message_size=message_size,
                               warmup_iteration_type="0",
                               fabric="efa")

    server_read_wrs_after_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_wrs")
    server_read_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "rdma_read_bytes")

    bytes_read = server_read_bytes_after_test - server_read_bytes_before_test
    # Check that the server RDMA-read the whole payload. The read lands
    # directly in the CUDA buffer, so gdrcopy and dmabuf registration do not
    # change the count. Fabtests control messages sent through libfabric can
    # add reads, so allow more than the payload.
    assert bytes_read >= message_size
    assert server_read_wrs_after_test >= server_read_wrs_before_test + 1
