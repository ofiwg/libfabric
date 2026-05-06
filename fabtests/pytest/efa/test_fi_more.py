import pytest
from common import ClientServerTest
from efa.efa_common import efa_run_client_server_test


@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["msg", "senddata", "write", "writedata"])
def test_fi_more_mt(cmdline_args, operation_type, fabric):
    cmd = "fi_efa_fi_more_mt -o " + operation_type
    test = ClientServerTest(cmdline_args, cmd, fabric=fabric, timeout=30)
    test.run()

@pytest.mark.functional
def test_rdm_tagged_bw_use_fi_more(cmdline_args, completion_semantic, memory_type, message_size):
    efa_run_client_server_test(cmdline_args, "fi_rdm_tagged_bw --use-fi-more",
                               "short", completion_semantic, memory_type, message_size, fabric="efa")

@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["writedata", "write"])
@pytest.mark.parametrize("iteration_type",
                         ["5", # smaller than max batch wqe cnt (16)
                          "48", # larger than max batch wqe cnt
                          "128"]) # larger than window size (64)
def test_rma_bw_use_fi_more(cmdline_args, operation_type, iteration_type, rma_bw_completion_semantic, inject_message_size, direct_rma_size, rma_fabric):
    command = "fi_rma_bw -e rdm -j 0 --use-fi-more"
    command = command + " -o " + operation_type
    # rma_bw test with data verification takes longer to finish
    timeout = max(540, cmdline_args.timeout)
    efa_run_client_server_test(cmdline_args, command, iteration_type, rma_bw_completion_semantic,
                               "host_to_host", direct_rma_size if rma_fabric == "efa-direct" else inject_message_size,
                               timeout=timeout, fabric=rma_fabric)