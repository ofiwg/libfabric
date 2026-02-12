import pytest
from common import ClientServerTest


@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["send", "senddata", "write", "writedata"])
def test_fi_more_mt(cmdline_args, operation_type, fabric):
    cmd = "fi_efa_fi_more_mt -o " + operation_type
    test = ClientServerTest(cmdline_args, cmd, fabric=fabric, timeout=30)
    test.run()
