import pytest
import copy
from common import UnitTest, has_cuda

@pytest.mark.unit
def test_mr_host(cmdline_args):
    test = UnitTest(cmdline_args, "fi_mr_test")
    test.run()


@pytest.mark.unit
def test_mr_cuda(cmdline_args):
    if not has_cuda(cmdline_args.server_id):
        pytest.skip("no cuda device")

    cmdline_args_copy = copy.copy(cmdline_args)
    cmdline_args_copy.append_environ("FI_EFA_USE_DEVICE_RDMA=1")

    test = UnitTest(cmdline_args_copy, "fi_mr_test -D cuda", check_warning=True)
    test.run()
