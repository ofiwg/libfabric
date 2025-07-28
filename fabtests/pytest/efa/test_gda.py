import os
import pytest
from common import ClientServerTest, has_cuda


@pytest.mark.short
@pytest.mark.functional
@pytest.mark.cuda_memory
def test_gda_send_recv(cmdline_args, direct_message_size, fabric):
    binpath = cmdline_args.binpath or ""
    if not os.path.exists(os.path.join(binpath, "fi_efa_gda")):
        pytest.skip("fi_efa_gda is not built")

    if not cmdline_args.do_dmabuf_reg_for_hmem:
        pytest.skip("DMABUF is required for GDA tests")

    if not has_cuda(cmdline_args.client_id) or not has_cuda(cmdline_args.server_id):
        pytest.skip("Client and server both need a cuda device")

    if fabric != "efa-direct":
        pytest.skip("GDA only works for efa-direct fabric")

    test = ClientServerTest(cmdline_args, "fi_efa_gda",
                            message_size=direct_message_size,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=fabric)
    test.run()
