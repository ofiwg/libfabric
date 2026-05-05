import pytest
from common import ClientServerTest

@pytest.fixture(params=[4096, 65536, 131072, 1048576])
def remote_exit_early_message_size(request, fabric):
    # efa-direct only supports up to MTU size
    if fabric == "efa-direct" and request.param != 4096:
        pytest.skip("efa-direct only supports 4K message size")
    return request.param

@pytest.mark.functional
def test_remote_exit_early_post_send(cmdline_args, fabric, remote_exit_early_message_size):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early",
                            message_size=remote_exit_early_message_size, fabric=fabric)
    test.run()

# efa-direct doesn't support tagged
@pytest.mark.functional
def test_remote_exit_early_post_tagged(cmdline_args, remote_exit_early_message_size):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early -o tagged",
                            message_size=remote_exit_early_message_size, fabric="efa")
    test.run()

@pytest.mark.functional
def test_remote_exit_early_post_writedata(cmdline_args, rma_fabric, remote_exit_early_message_size):
    test = ClientServerTest(cmdline_args, "fi_efa_rdm_remote_exit_early -o writedata",
                            message_size=remote_exit_early_message_size, fabric=rma_fabric)
    test.run()

@pytest.mark.functional
def test_remote_exit_early_post_rx(cmdline_args, fabric, remote_exit_early_message_size):
    test = ClientServerTest(cmdline_args,
                            "fi_efa_rdm_remote_exit_early --post-rx",
                            message_size=remote_exit_early_message_size, fabric=fabric)
    test.run()
