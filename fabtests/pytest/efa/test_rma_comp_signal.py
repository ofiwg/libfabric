import pytest
from common import ClientServerTest


# Completion with signal is an EFA-direct feature; run only on efa-direct.
# The test itself reports ENODATA (a skip) when the device/driver does not
# advertise completion-with-signal support.
@pytest.mark.comp_signal
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.functional
@pytest.mark.parametrize("signal_mode", ["local", "remote", "both"])
@pytest.mark.parametrize("signal_width", ["8", "16", "32"])
def test_efa_rma_comp_signal(cmdline_args, signal_mode, signal_width):
    command = ("fi_efa_rma_comp_signal"
               " --signal-mode " + signal_mode +
               " --signal-width " + signal_width +
               " -I 8")
    test = ClientServerTest(cmdline_args, command,
                            iteration_type=None,
                            message_size=None,
                            fabric="efa-direct")
    test.run()
