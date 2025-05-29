import pytest
from common import ClientServerTest

@pytest.mark.functional
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


# TODO: expand to different meory types
@pytest.mark.unstable
@pytest.mark.parametrize("msg_size", [16, 8192, 16384, 1048576])
@pytest.mark.parametrize("shared_av", [True, False])
def test_multi_ep_mt(cmdline_args, msg_size, shared_av):
    cmd = f"fi_efa_multi_ep_mt -e rdm -c 20 -I 100"

    if shared_av:
        cmd += " -A"

    test = ClientServerTest(cmdline_args, cmd, message_size=msg_size, fabric="efa", timeout=30, additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()
