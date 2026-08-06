import pytest

from common import (
    client_server_have_device,
    test_selected_by_marker,
)

# Memory types that yield a distinct permutation for a bi-directional
# transfer. A bi-directional test moves data both ways, so <a>_to_<b> and
# <b>_to_<a> drive the same code paths and only one of the pair is kept.
memory_type_list_bi_dir = [
    pytest.param("host_to_host"),
    pytest.param("host_to_cuda", marks=pytest.mark.cuda_memory),
    pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory),
]

# Every memory type. Only a uni-directional test needs cuda_to_host, which for
# a bi-directional transfer is host_to_cuda with the endpoint roles swapped.
memory_type_list = memory_type_list_bi_dir + [
    pytest.param("cuda_to_host", marks=pytest.mark.cuda_memory),
]


@pytest.fixture(scope="module", params=["read", "writedata", "write"])
def rma_operation_type(request):
    return request.param


@pytest.fixture(scope="module")
def rma_bw_memory_type(memory_type, rma_operation_type):
    is_test_bi_dir = rma_operation_type != "writedata"
    if is_test_bi_dir and (memory_type not in [_.values[0] for _ in memory_type_list_bi_dir]):
        pytest.skip("Duplicated memory type for bi-directional test")
    return memory_type


@pytest.fixture(scope="function")
def rma_bw_completion_semantic(completion_semantic, rma_operation_type):
    if completion_semantic != "delivery_complete" and rma_operation_type == "read":
        # A read is not a transmission, so there is no difference between the
        # delivery_complete and transmit_complete semantics.
        pytest.skip("Duplicate completion semantic for fi_read test")
    return completion_semantic


def add_memory_type_parametrization(metafunc):
    # Drop permutations whose device is absent on the owning endpoint so they
    # are never collected. Fall back to all candidates when ids are absent or
    # detection fails; the runtime skip in common.py is the safety net.
    if "memory_type" not in metafunc.fixturenames:
        return
    server_id = metafunc.config.getoption("--server-id", default=None)
    client_id = metafunc.config.getoption("--client-id", default=None)
    if not server_id or not client_id:
        params = memory_type_list
    else:
        try:
            params = [p for p in memory_type_list
                      if client_server_have_device(p.values[0], server_id, client_id)]
        except Exception:
            params = memory_type_list
    metafunc.parametrize("memory_type", params, scope="module")


def pytest_generate_tests(metafunc):
    add_memory_type_parametrization(metafunc)

    if "message_sizes" not in metafunc.fixturenames:
        return

    marker = next(metafunc.definition.iter_markers("message_sizes"), None)
    if marker is None:
        return

    test_markers = {m.name for m in metafunc.definition.iter_markers()}
    is_pr_ci = test_selected_by_marker(metafunc.config, test_markers, "pr_ci")
    default = marker.kwargs["default"]
    pr_ci = marker.kwargs.get("pr_ci", default)

    metafunc.parametrize("message_sizes", pr_ci if is_pr_ci else default)
