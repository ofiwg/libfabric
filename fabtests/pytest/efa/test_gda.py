import os
import pytest
from common import ClientServerTest, has_cuda
from efa.efa_common import DIRECT_SIZES, DIRECT_RMA_SIZES

pytestmark = [
    pytest.mark.short,
    pytest.mark.functional,
    pytest.mark.cuda_memory,
    pytest.mark.fabric(params=["efa-direct"]),
]


@pytest.fixture(autouse=True)
def skip_if_gda_unavailable(cmdline_args):
    binpath = cmdline_args.binpath or ""
    if not os.path.exists(os.path.join(binpath, "fi_efa_gda")):
        pytest.skip("fi_efa_gda is not built")

    if not cmdline_args.do_dmabuf_reg_for_hmem:
        pytest.skip("DMABUF is required for GDA tests")

    if not has_cuda(cmdline_args.client_id) or not has_cuda(cmdline_args.server_id):
        pytest.skip("Client and server both need a cuda device")


@pytest.fixture()
def skip_if_hw_cntr_unavailable(cmdline_args):
    binpath = cmdline_args.binpath or ""
    if not os.path.exists(os.path.join(binpath, "fi_efa_hw_cntr")):
        pytest.skip("hw cntr tests require efadv_create_comp_cntr")


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
@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES, pr_ci_efa_direct=DIRECT_RMA_SIZES)
def test_gda_write_bw(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o write",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            datacheck_type="with_datacheck")
    test.run()


@pytest.mark.pr_ci
@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES, pr_ci_efa_direct=DIRECT_RMA_SIZES)
def test_gda_writedata_bw(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o writedata",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            datacheck_type="with_datacheck")
    test.run()


@pytest.mark.pr_ci
@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES, pr_ci_efa_direct=DIRECT_RMA_SIZES)
def test_gda_read_bw(cmdline_args, message_sizes, rma_fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o read",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            datacheck_type="with_datacheck")
    test.run()


@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES)
def test_gda_send_recv_hw_cntr(cmdline_args, message_sizes, fabric, skip_if_hw_cntr_unavailable):
    test = ClientServerTest(cmdline_args, "fi_efa_gda --use-hw-cntr",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()


@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES)
def test_gda_write_bw_hw_cntr(cmdline_args, message_sizes, rma_fabric, skip_if_hw_cntr_unavailable):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o write --use-hw-cntr",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()


@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES)
def test_gda_writedata_bw_hw_cntr(cmdline_args, message_sizes, rma_fabric, skip_if_hw_cntr_unavailable):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o writedata --use-hw-cntr",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()


@pytest.mark.message_sizes(default_efa_direct=DIRECT_RMA_SIZES)
def test_gda_read_bw_hw_cntr(cmdline_args, message_sizes, rma_fabric, skip_if_hw_cntr_unavailable):
    test = ClientServerTest(cmdline_args, "fi_efa_gda -o read --use-hw-cntr",
                            message_size=message_sizes,
                            iteration_type="short",
                            memory_type="cuda_to_cuda",
                            fabric=rma_fabric,
                            additional_env="FI_EFA_USE_HW_CNTR=1")
    test.run()
