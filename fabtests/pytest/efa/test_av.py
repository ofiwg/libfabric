import pytest

@pytest.mark.functional
def test_av_xfer(cmdline_args, fabric):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, "fi_av_xfer -e rdm", fabric=fabric)
    test.run()
