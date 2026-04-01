import pytest
from common import ClientServerTest


@pytest.mark.functional
@pytest.mark.parametrize("rma_op", ["write", "read", "writedata"])
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
def test_mr_abort(cmdline_args, fabric, rma_op, cancel_order, close_side):
    if close_side == "target" and rma_op == "writedata":
        pytest.skip("target-close uses write-with-imm signaling, incompatible with writedata op")
    command = (f"fi_mr_abort -e rdm -T abort -o {rma_op} -C {cancel_order}"
               f" -R {close_side} -W 8192 -S 1048576")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()


@pytest.mark.functional
@pytest.mark.parametrize("rma_op", ["write", "read", "writedata"])
def test_mr_abort_multi_op_per_mr(cmdline_args, fabric, rma_op):
    command = (f"fi_mr_abort -e rdm -T abort -o {rma_op}"
               f" -N 4 -C random -W 8192 -S 1048576")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()


@pytest.mark.functional
@pytest.mark.parametrize("rma_op", ["write", "read", "writedata"])
def test_mr_abort_partial(cmdline_args, fabric, rma_op):
    command = f"fi_mr_abort -e rdm -T partial -o {rma_op} -S 10485760"
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()


@pytest.mark.functional
@pytest.mark.parametrize("message_size", [4096, 65536, 262144])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
def test_mr_abort_send(cmdline_args, fabric, message_size, close_side):
    if fabric == "efa-direct" and message_size > 8192:
        pytest.skip("efa-direct max send size is 8KB")
    command = (f"fi_mr_abort -e rdm -T send -S {message_size}"
               f" -R {close_side} -W 8192")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()


@pytest.mark.functional
@pytest.mark.parametrize("message_size", [4096, 65536, 262144])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
def test_mr_abort_tagged(cmdline_args, fabric, message_size, close_side):
    if fabric == "efa-direct":
        pytest.skip("efa-direct does not support tagged messages")
    command = (f"fi_mr_abort -e rdm -T tagged -S {message_size}"
               f" -R {close_side} -W 8192")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric)
    test.run()
