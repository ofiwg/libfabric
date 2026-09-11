import pytest
from common import ClientServerTest


# Completion action is an EFA-direct feature; run only on efa-direct.
# The test itself reports ENODATA (a skip) when the device/driver does not
# advertise completion-action support.
@pytest.mark.comp_action
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.functional
@pytest.mark.parametrize("action_mode", ["local", "remote", "both"])
@pytest.mark.parametrize("action_width", ["8", "16", "32"])
def test_efa_rma_comp_action(cmdline_args, action_mode, action_width):
    command = ("fi_efa_rma_comp_action"
               " --action-mode " + action_mode +
               " --action-width " + action_width +
               " -I 8")
    test = ClientServerTest(cmdline_args, command,
                            iteration_type=None,
                            message_size=None,
                            fabric="efa-direct")
    test.run()
