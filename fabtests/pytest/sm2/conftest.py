import pytest

from common import client_server_have_device

memory_type_list = [
    pytest.param("host_to_host"),
    pytest.param("host_to_cuda", marks=pytest.mark.cuda_memory),
    pytest.param("cuda_to_host", marks=pytest.mark.cuda_memory),
    pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory),
]


def pytest_generate_tests(metafunc):
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
