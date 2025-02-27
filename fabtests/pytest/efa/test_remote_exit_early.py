import pytest
from common import ClientServerTest

@pytest.fixture(params=[65536, 131072, 1048576])
def remote_exit_early_message_size(request):
    # 64K use medium
    # 128K use long CTS
    # 1M use runtread or longread if rdma read is available
    return request.param

@pytest.mark.functional
def test_remote_exit_early_post_send(cmdline_args, remote_exit_early_message_size):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early",
                            message_size=remote_exit_early_message_size)
    test.run()

@pytest.mark.functional
def test_remote_exit_early_post_tagged(cmdline_args, remote_exit_early_message_size):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early -o tagged",
                            message_size=remote_exit_early_message_size)
    test.run()

@pytest.mark.functional
def test_remote_exit_early_post_writedata(cmdline_args, remote_exit_early_message_size):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early -o writedata",
                            message_size=remote_exit_early_message_size)
    test.run()

@pytest.mark.functional
def test_remote_exit_early_post_rx(cmdline_args, remote_exit_early_message_size):
    test = ClientServerTest(cmdline_args,
                            "fi_efa_rdm_remote_exit_early --post-rx",
                            message_size=remote_exit_early_message_size)
    test.run()
