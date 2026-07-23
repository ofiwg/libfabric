import pytest
from common import ClientServerTest
from efa.efa_common import REMOTE_EXIT_SIZES, REMOTE_EXIT_SIZES_DIRECT

# 64K use medium
# 128K use long CTS
# 1M use runtread or longread if rdma read is available

@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=REMOTE_EXIT_SIZES, default_efa_direct=REMOTE_EXIT_SIZES_DIRECT,
                           pr_ci_efa=REMOTE_EXIT_SIZES, pr_ci_efa_direct=REMOTE_EXIT_SIZES_DIRECT)
@pytest.mark.functional
def test_remote_exit_early_post_send(cmdline_args, fabric, message_sizes):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early",
                            message_size=message_sizes, fabric=fabric)
    test.run()

# efa-direct doesn't support tagged
@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa"])
@pytest.mark.message_sizes(default_efa=REMOTE_EXIT_SIZES, pr_ci_efa=REMOTE_EXIT_SIZES)
@pytest.mark.functional
def test_remote_exit_early_post_tagged(cmdline_args, fabric, message_sizes):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early -o tagged",
                            message_size=message_sizes, fabric=fabric)
    test.run()

@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=REMOTE_EXIT_SIZES, default_efa_direct=REMOTE_EXIT_SIZES_DIRECT,
                           pr_ci_efa=REMOTE_EXIT_SIZES, pr_ci_efa_direct=REMOTE_EXIT_SIZES_DIRECT)
@pytest.mark.functional
def test_remote_exit_early_post_writedata(cmdline_args, rma_fabric, message_sizes):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early -o writedata",
                            message_size=message_sizes, fabric=rma_fabric)
    test.run()

@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.message_sizes(default_efa=REMOTE_EXIT_SIZES, default_efa_direct=REMOTE_EXIT_SIZES_DIRECT,
                           pr_ci_efa=REMOTE_EXIT_SIZES, pr_ci_efa_direct=REMOTE_EXIT_SIZES_DIRECT)
@pytest.mark.functional
def test_remote_exit_early_post_rx(cmdline_args, fabric, message_sizes):
    test = ClientServerTest(cmdline_args,
                            "fi_efa_rdm_remote_exit_early --post-rx",
                            message_size=message_sizes, fabric=fabric)
    test.run()
