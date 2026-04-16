import pytest
from common import ClientServerTest
from efa.efa_common import REMOTE_EXIT_SIZES

# 64K use medium
# 128K use long CTS
# 1M use runtread or longread if rdma read is available

@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=REMOTE_EXIT_SIZES)
@pytest.mark.functional
def test_remote_exit_early_post_send(cmdline_args, message_sizes):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early",
                            message_size=message_sizes)
    test.run()

@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=REMOTE_EXIT_SIZES)
@pytest.mark.functional
def test_remote_exit_early_post_tagged(cmdline_args, message_sizes):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early -o tagged",
                            message_size=message_sizes)
    test.run()

@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=REMOTE_EXIT_SIZES)
@pytest.mark.functional
def test_remote_exit_early_post_writedata(cmdline_args, message_sizes):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early -o writedata",
                            message_size=message_sizes)
    test.run()

@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=REMOTE_EXIT_SIZES)
@pytest.mark.functional
def test_remote_exit_early_post_rx(cmdline_args, message_sizes):
    test = ClientServerTest(cmdline_args,
                            "fi_efa_rdm_remote_exit_early --post-rx",
                            message_size=message_sizes)
    test.run()
