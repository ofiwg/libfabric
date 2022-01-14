import pytest

@pytest.mark.functional
def test_unexpected_msg(cmdline_args):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, "fi_unexpected_msg -e rdm -I 10")
    test.run()
