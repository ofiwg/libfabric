import pytest

@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rma_bw(cmdline_args, iteration_type, operation_type, completion_type):
    from common import ClientServerTest
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + operation_type
    test = ClientServerTest(cmdline_args, command, iteration_type, completion_type, message_size="all")
    test.run()

@pytest.mark.functional
@pytest.mark.parametrize("operation_type", ["read", "writedata", "write"])
def test_rma_bw_range(cmdline_args, operation_type, completion_type, message_size):
    from common import ClientServerTest
    command = "fi_rma_bw -e rdm"
    command = command + " -o " + operation_type
    test = ClientServerTest(cmdline_args, command, iteration_type="short", completion_type=completion_type, message_size=message_size)
    test.run()
