import pytest
from common import ClientServerTest

@pytest.mark.functional
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.parametrize("shared_cq", [True, False])
@pytest.mark.parametrize("shared_av", [True, False])
def test_multi_ep(cmdline_args, shared_cq, shared_av, rma_fabric):
    # This test requests FI_RMA
    cmd = "fi_multi_ep -e rdm"
    if shared_cq:
        cmd += " -Q"
    if shared_av:
        cmd += " -A"
    test = ClientServerTest(cmdline_args, cmd, message_size=256, fabric=rma_fabric)
    test.run()

