import pytest

@pytest.mark.functional
@pytest.mark.parametrize("shared_cq", [True, False])
def test_multi_ep(cmdline_args, shared_cq, rma_fabric):
    # This test requests FI_RMA
    from common import ClientServerTest
    cmd = "fi_multi_ep -e rdm"
    if shared_cq:
        cmd += "  -Q"
    test = ClientServerTest(cmdline_args, cmd, message_size=256, fabric=rma_fabric)
    test.run()
