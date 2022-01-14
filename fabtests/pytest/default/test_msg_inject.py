import pytest

@pytest.mark.functional
@pytest.mark.parametrize("api_type", ["sendmsg", "post_tx"])
@pytest.mark.parametrize("flag", ["inject", "inj_complete"])
def test_msg_inject(cmdline_args, api_type, flag):
    from common import ClientServerTest

    command = "fi_msg_inject"
    if api_type == "sendmsg":
        command += " -N"
    command += " -A " + flag
    test = ClientServerTest(cmdline_args, command)
    test.run()
