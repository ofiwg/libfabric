import pytest
from common import ClientServerTest

failing_test_skip = pytest.mark.skip(reason="efa/efa-direct fabrics do not currently pass fi_mr_abort testing")


# --- Test: abort (RMA) ---

@failing_test_skip
@pytest.mark.functional
@pytest.mark.parametrize("rma_op", ["write", "read", "writedata"])
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
@pytest.mark.parametrize("ops_per_mr", [1, 4])
def test_mr_abort(cmdline_args, fabric, rma_op, cancel_order, close_side, ops_per_mr):
    command = (f"fi_mr_abort -T abort -o {rma_op} -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W 4096"
               f" -S 10485760")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()


# --- Test: partial (2 MRs on same buffer) ---
@failing_test_skip
@pytest.mark.functional
@pytest.mark.parametrize("rma_op", ["write", "read", "writedata"])
def test_mr_abort_partial(cmdline_args, fabric, rma_op):
    command = (f"fi_mr_abort -T partial -o {rma_op} -S 10485760")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()


# --- Test: send ---

@failing_test_skip
@pytest.mark.functional
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
@pytest.mark.parametrize("ops_per_mr", [1, 4])
@pytest.mark.parametrize("message_size", [4096, 65536, 262144])
def test_mr_abort_send(cmdline_args, fabric, cancel_order, close_side,
                       ops_per_mr, message_size):
    if fabric == "efa-direct" and message_size > 8192:
        pytest.skip("efa-direct max send size is 8KB")

    if close_side == "target":
        pytest.skip("efa does not currently support canceling posted RX buffers")

    command = (f"fi_mr_abort -T send -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W 4096"
               f" -S {message_size}  -D ep_first")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()


# --- Test: tagged ---

@failing_test_skip
@pytest.mark.functional
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
@pytest.mark.parametrize("ops_per_mr", [1, 4])
@pytest.mark.parametrize("message_size", [4096, 65536, 262144])
def test_mr_abort_tagged(cmdline_args, fabric, cancel_order, close_side,
                         ops_per_mr, message_size):
    if fabric == "efa-direct":
        pytest.skip("efa-direct does not support tagged messages")

    if close_side == "target":
        pytest.skip("efa does not currently support canceling posted RX buffers")

    command = (f"fi_mr_abort -T tagged -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W 4096"
               f" -S {message_size} -D ep_first")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()
