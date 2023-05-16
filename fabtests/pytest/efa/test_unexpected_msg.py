import pytest

@pytest.mark.functional
def test_unexpected_msg(cmdline_args):
    from common import ClientServerTest
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("Temporarily skip single node before the unknown peer bug is fixed")
    test = ClientServerTest(cmdline_args, "fi_unexpected_msg -e rdm -I 10")
    test.run()
