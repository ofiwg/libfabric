import copy

import pytest
from common import UnitTest, has_cuda, has_neuron


@pytest.mark.unit
def test_mr_host(cmdline_args):
    test = UnitTest(cmdline_args, "fi_mr_test")
    test.run()


@pytest.mark.unit
@pytest.mark.parametrize(
    "hmem_type",
    [
        pytest.param("cuda", marks=pytest.mark.cuda_memory),
        pytest.param("neuron", marks=pytest.mark.neuron_memory),
    ],
)
def test_mr_hmem(cmdline_args, hmem_type):
    if hmem_type == "cuda" and not has_cuda(cmdline_args.server_id):
        pytest.skip("no cuda device")
    if hmem_type == "neuron" and not has_neuron(cmdline_args.server_id):
        pytest.skip("no neuron device")

    cmdline_args_copy = copy.copy(cmdline_args)

    test = UnitTest(
        cmdline_args_copy,
        f"fi_mr_test -D {hmem_type}",
        failing_warn_msgs=["Unable to add MR to map"],
    )
    test.run()
