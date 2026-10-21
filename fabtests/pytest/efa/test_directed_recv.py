import pytest
from efa.efa_common import efa_run_client_server_test, memory_type_list_all, memory_type_list_device_to_device


# efa-direct does not support FI_DIRECTED_RECV
@pytest.mark.pr_ci
@pytest.mark.pr_ci_hmem
@pytest.mark.functional
@pytest.mark.parametrize("op", ["msg", "tagged"])
@pytest.mark.parametrize("msg_size", [64, 16384, 1048576]) # eager, medium and long protocols
@pytest.mark.memory_type(memory_type_list_all, pr_ci_hmem=memory_type_list_device_to_device)
def test_directed_recv(cmdline_args, op, msg_size, memory_type, completion_semantic):
    efa_run_client_server_test(cmdline_args, f"fi_directed_recv -e rdm -o {op}", iteration_type=None,
                               completion_semantic=completion_semantic, memory_type=memory_type,
                               message_size=msg_size, fabric="efa")
