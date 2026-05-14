import copy

import pytest
from common import UnitTest, has_cuda, has_neuron


@pytest.mark.pr_ci
@pytest.mark.unit
def test_mr_host(cmdline_args):
    test = UnitTest(cmdline_args, "fi_mr_test")
    test.run()


@pytest.mark.pr_ci
@pytest.mark.unit
@pytest.mark.short
def test_mr_hmem(cmdline_args, hmem_type, fabric):
    if hmem_type == "cuda" and not has_cuda(cmdline_args.server_id):
        pytest.skip("no cuda device")
    if hmem_type == "neuron" and not has_neuron(cmdline_args.server_id):
        pytest.skip("no neuron device")

    cmdline_args_copy = copy.copy(cmdline_args)

    test_command = f"fi_mr_test -D {hmem_type} -f {fabric}"

    if cmdline_args.do_dmabuf_reg_for_hmem:
        test_command += " -R"

    test = UnitTest(
        cmdline_args_copy,
        test_command,
        failing_warn_msgs=["Unable to add MR to map"],
    )
    test.run()


@pytest.mark.unit
@pytest.mark.short
def test_efa_mr_hmem(cmdline_args, hmem_type, fabric):
    if hmem_type != "neuron":
        pytest.skip("test only applies to neuron")
    if not has_neuron(cmdline_args.server_id):
        pytest.skip("no neuron device")
    if not cmdline_args.do_dmabuf_reg_for_hmem:
        pytest.skip("This test tests neuron get_dmabuf_fd_vX and needs dmabuf to be enabled and working to test these apis")

    cmdline_args_copy = copy.copy(cmdline_args)

    test_command = f"fi_efa_mr_test -D neuron -f {fabric}"

    if cmdline_args.do_dmabuf_reg_for_hmem:
        test_command += " -R"

    test = UnitTest(cmdline_args_copy, test_command)
    test.run()
