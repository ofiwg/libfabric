import pytest

from common import test_selected_by_marker, num_hmem_devices

memory_type_list = [
    pytest.param("host_to_host"),
    pytest.param("host_to_cuda", marks=pytest.mark.cuda_memory),
    pytest.param("cuda_to_host", marks=pytest.mark.cuda_memory),
    pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory),
]


def _endpoint_has_device(memory_token, server_id, client_id):
    client_memory_type, server_memory_type = memory_token.split("_to_")
    for memory_type_name, ip in ((client_memory_type, client_id),
                                 (server_memory_type, server_id)):
        if memory_type_name == "host":
            continue
        if num_hmem_devices(ip, memory_type_name) <= 0:
            return False
    return True


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
                      if _endpoint_has_device(p.values[0], server_id, client_id)]
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
