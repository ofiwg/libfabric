import pytest

@pytest.mark.functional
@pytest.mark.parametrize("shared_cq", [True, False])
def test_multi_ep_cq(cmdline_args, shared_cq):
    from common import ClientServerTest
    cmd = "fi_multi_ep -e rdm"
    if shared_cq:
        cmd += "  -Q"
    test = ClientServerTest(cmdline_args, cmd)
    test.run()

@pytest.mark.functional
def test_multi_ep_av(cmdline_args):
    from common import ClientServerTest
    cmd = "fi_multi_ep -e rdm -A"
    test = ClientServerTest(cmdline_args, cmd)
    test.run()
