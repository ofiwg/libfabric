import os
import pytest
from common import ClientServerTest, has_cuda


@pytest.mark.functional
@pytest.mark.cuda_memory
def test_gda_send_recv(cmdline_args, direct_message_size):
    fi_efa_gda_path = "fi_efa_gda"
    if cmdline_args.binpath:
        fi_efa_gda_path = os.path.join(cmdline_args.binpath, "fi_efa_gda")
    if not os.path.exists(fi_efa_gda_path):
        pytest.skip("fi_efa_gda is not built")

    if not cmdline_args.do_dmabuf_reg_for_hmem:
        pytest.skip("DMABUF is required for GDA tests")

    if not has_cuda(cmdline_args.client_id) or not has_cuda(cmdline_args.server_id):
        pytest.skip("Client and server both need a cuda device")

    test = ClientServerTest(cmdline_args, "fi_efa_gda",
                            message_size=direct_message_size,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric="efa-direct")
    test.run()
