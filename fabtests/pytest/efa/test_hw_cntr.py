import pytest
from common import ClientServerTest
from efa.efa_common import DIRECT_SIZES, DIRECT_RMA_SIZES

@pytest.mark.hw_cntr
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES)
@pytest.mark.short
@pytest.mark.functional
def test_efa_hw_cntr_pingpong(cmdline_args, message_sizes, fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_hw_cntr",
                            iteration_type="short",
                            message_size=message_sizes,
                            fabric=fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()

@pytest.mark.hw_cntr
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES)
@pytest.mark.short
@pytest.mark.functional
def test_efa_hw_cntr_rma_write(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_hw_cntr -o write",
                            iteration_type="short",
                            message_size=message_sizes,
                            fabric=rma_fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()

@pytest.mark.hw_cntr
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES)
@pytest.mark.short
@pytest.mark.functional
def test_efa_hw_cntr_pingpong_ext_mem(cmdline_args, message_sizes, fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_hw_cntr --external-mem",
                            iteration_type="short",
                            message_size=message_sizes,
                            fabric=fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()

@pytest.mark.hw_cntr
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES)
@pytest.mark.short
@pytest.mark.functional
def test_efa_hw_cntr_rma_write_ext_mem(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_hw_cntr --external-mem -o write",
                            iteration_type="short",
                            message_size=message_sizes,
                            fabric=rma_fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()
