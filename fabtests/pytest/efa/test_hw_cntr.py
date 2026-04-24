import os
import pytest
from common import ClientServerTest
from efa.efa_common import DIRECT_SIZES

@pytest.fixture(autouse=True)
def skip_if_not_built(cmdline_args):
    binpath = cmdline_args.binpath or ""
    if not os.path.exists(os.path.join(binpath, "fi_efa_hw_cntr")):
        pytest.skip("fi_efa_hw_cntr requires efadv_create_comp_cntr")


@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.message_sizes(default_efa_direct=DIRECT_SIZES)
@pytest.mark.short
@pytest.mark.functional
def test_efa_hw_cntr_pingpong(cmdline_args, message_sizes, fabric):
    test = ClientServerTest(cmdline_args, "fi_efa_hw_cntr",
                            iteration_type="short",
                            message_size=message_sizes,
                            fabric=fabric)
    test.run()
