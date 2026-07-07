import pytest

# shm only supports RDM EPs
@pytest.mark.functional
@pytest.mark.parametrize("endpoint_type", ["rdm"])
def test_av_xfer(cmdline_args, endpoint_type):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, "fi_av_xfer -e " + endpoint_type)
    test.run()
