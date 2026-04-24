import os
import pytest
from common import ClientServerTest

def _skip_if_not_built(cmdline_args):
    binpath = cmdline_args.binpath or ""
    if not os.path.exists(os.path.join(binpath, "fi_efa_hw_cntr")):
        pytest.skip("fi_efa_hw_cntr requires efadv_create_comp_cntr")

@pytest.mark.functional
def test_efa_hw_cntr_pingpong(cmdline_args, direct_message_size, fabric):
    _skip_if_not_built(cmdline_args)
    if fabric != "efa-direct":
        pytest.skip("hw_cntr is only in efa-direct")
    test = ClientServerTest(cmdline_args, "fi_efa_hw_cntr",
                            iteration_type="short",
                            message_size=direct_message_size,
                            fabric=fabric)
    test.run()


@pytest.mark.functional
def test_efa_hw_cntr_rma_write(cmdline_args, direct_rma_size, rma_fabric):
    _skip_if_not_built(cmdline_args)
    if rma_fabric != "efa-direct":
        pytest.skip("hw_cntr is only in efa-direct")
    test = ClientServerTest(cmdline_args, "fi_efa_hw_cntr -o write",
                            iteration_type="short",
                            message_size=direct_rma_size,
                            fabric=rma_fabric)
    test.run()
