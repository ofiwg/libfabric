import pytest
from efa.efa_common import has_rdma

# this test must be run in serial mode because it will open the maximal number
# of cq that efa device can support
@pytest.mark.serial
@pytest.mark.unit
def test_cq(cmdline_args, fabric):
    from common import UnitTest
    test = UnitTest(cmdline_args, f"fi_cq_test -f {fabric}")
    test.run()

@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["senddata", "writedata"])
def test_cq_data(cmdline_args, operation_type, fabric):
    from common import ClientServerTest
    if fabric == "efa-direct" and operation_type == "writedata" and not has_rdma(cmdline_args, operation_type):
        pytest.skip("FI_RMA is not supported. Skip writedata test on efa-direct.")
    test = ClientServerTest(cmdline_args, f"fi_cq_data -e rdm -o " + operation_type, fabric=fabric)
    test.run()
