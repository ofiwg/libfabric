import os
import pytest
from common import ClientServerTest, has_cuda
from efa.efa_common import DIRECT_SIZES


@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES, pr_ci_efa_direct=DIRECT_SIZES)
@pytest.mark.short
@pytest.mark.functional
@pytest.mark.cuda_memory
def test_gda_send_recv(cmdline_args, message_sizes, fabric):
    binpath = cmdline_args.binpath or ""
    if not os.path.exists(os.path.join(binpath, "fi_efa_gda")):
        pytest.skip("fi_efa_gda is not built")

    if not cmdline_args.do_dmabuf_reg_for_hmem:
        pytest.skip("DMABUF is required for GDA tests")

    if not has_cuda(cmdline_args.client_id) or not has_cuda(cmdline_args.server_id):
        pytest.skip("Client and server both need a cuda device")

    test = ClientServerTest(cmdline_args, "fi_efa_gda",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=fabric)
    test.run()
