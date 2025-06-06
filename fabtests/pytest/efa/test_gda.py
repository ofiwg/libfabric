import pytest
from common import ClientServerTest, has_cuda


@pytest.mark.functional
@pytest.mark.cuda_memory
def test_gda_send_recv(cmdline_args, direct_message_size):
    if not has_cuda(cmdline_args.client_id) or not has_cuda(cmdline_args.server_id):
        pytest.skip("Client and server both need a cuda device")

    cmdline_args.do_dmabuf_reg_for_hmem = True

    test = ClientServerTest(cmdline_args, "fi_efa_gda",
                            datacheck_type="with_datacheck",
                            message_size=direct_message_size,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric="efa-direct")
    test.run()
