import pytest

from common import test_selected_by_marker


def pytest_generate_tests(metafunc):
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


@pytest.fixture(scope="module", params=["host_to_host",
                                        pytest.param("host_to_cuda", marks=pytest.mark.cuda_memory),
                                        pytest.param("cuda_to_host", marks=pytest.mark.cuda_memory),
                                        pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory)])
def memory_type(request):
    return request.param
