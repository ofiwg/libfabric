import pytest
from common import MULTI_RECV_SIZES


@pytest.mark.pr_ci
@pytest.mark.message_sizes(default=MULTI_RECV_SIZES)
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
# efa-direct does not support multi-recv
def test_multi_recv(cmdline_args, iteration_type, message_sizes):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args,
            "fi_multi_recv -e rdm",
            iteration_type,
            message_size=message_sizes,
            fabric="efa")
    test.run()
