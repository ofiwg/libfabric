import pytest

from default.test_rdm import test_rdm_bw_functional
from default.test_rdm import test_rdm_atomic

def run_rdm_test(cmdline_args, executable, iteration_type, completion_type, memory_type, message_size_range):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, executable, iteration_type,
                            completion_type=completion_type,
                            datacheck_type="with_datacheck",
                            message_size_range=message_size_range,
                            memory_type=memory_type)
    test.run()

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rdm_pingpong(cmdline_args, iteration_type, completion_type, memory_type):
    run_rdm_test(cmdline_args, "fi_rdm_pingpong", iteration_type,
            completion_type, memory_type, "all")

@pytest.mark.functional
def test_rdm_pingpong_range(cmdline_args, completion_type, memory_type, message_size_range):
    run_rdm_test(cmdline_args, "fi_rdm_pingpong", "short",
                 completion_type, memory_type, message_size_range)

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rdm_tagged_pingpong(cmdline_args, iteration_type, completion_type, memory_type):
    run_rdm_test(cmdline_args, "fi_rdm_tagged_pingpong", iteration_type,
                 completion_type, memory_type, "all")

@pytest.mark.functional
def test_rdm_tagged_pingpong_range(cmdline_args, completion_type, memory_type, message_size_range):
    run_rdm_test(cmdline_args, "fi_rdm_tagged_pingpong", "short",
                 completion_type, memory_type, message_size_range)

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rdm_tagged_bw(cmdline_args, iteration_type, completion_type, memory_type):
    run_rdm_test(cmdline_args, "fi_rdm_tagged_bw", iteration_type,
                 completion_type, memory_type, "all")

@pytest.mark.functional
def test_rdm_tagged_bw_range(cmdline_args, completion_type, memory_type, message_size_range):
    run_rdm_test(cmdline_args, "fi_rdm_tagged_bw", "short",
                 completion_type, memory_type, message_size_range)
