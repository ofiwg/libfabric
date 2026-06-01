import pytest
from common import ClientServerTest, has_cuda
from efa.efa_common import DIRECT_SIZES, DIRECT_RMA_SIZES

# Apply to all functions in this file
pytestmark = [
    pytest.mark.short,
    pytest.mark.functional,
    pytest.mark.cuda_memory,
    pytest.mark.gda,
    pytest.mark.fabric(params=["efa-direct"]),
]


@pytest.mark.pr_ci
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES, pr_ci_efa_direct=DIRECT_SIZES)
def test_gda_send_recv(cmdline_args, message_sizes, fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=fabric,
                            datacheck_type="with_datacheck")
    test.run()


@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES, pr_ci_efa_direct=DIRECT_RMA_SIZES)
@pytest.mark.short
@pytest.mark.functional
@pytest.mark.cuda_memory
def test_gda_write_bw(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o write",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            datacheck_type="with_datacheck")
    test.run()


@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES, pr_ci_efa_direct=DIRECT_RMA_SIZES)
@pytest.mark.short
@pytest.mark.functional
@pytest.mark.cuda_memory
def test_gda_writedata_bw(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o writedata",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            datacheck_type="with_datacheck")
    test.run()


@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES, pr_ci_efa_direct=DIRECT_RMA_SIZES)
@pytest.mark.short
@pytest.mark.functional
@pytest.mark.cuda_memory
def test_gda_read_bw(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o read",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            datacheck_type="with_datacheck")
    test.run()


@pytest.mark.hw_cntr
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES)
def test_gda_send_recv_hw_cntr(cmdline_args, message_sizes, fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda --use-hw-cntr",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()


@pytest.mark.hw_cntr
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES)
def test_gda_write_bw_hw_cntr(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o write --use-hw-cntr",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()


@pytest.mark.hw_cntr
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES)
def test_gda_writedata_bw_hw_cntr(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o writedata --use-hw-cntr",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()


@pytest.mark.hw_cntr
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES)
def test_gda_read_bw_hw_cntr(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o read --use-hw-cntr",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()
