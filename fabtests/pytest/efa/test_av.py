import pytest

# This test skips efa-direct because it requests FI_TAGGED
@pytest.mark.functional
def test_av_xfer(cmdline_args):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, "fi_av_xfer -e rdm", fabric="efa")
    test.run()
