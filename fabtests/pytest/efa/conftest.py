import pytest

@pytest.fixture(scope="module", params=["host_to_host", "host_to_cuda",
                                        "cuda_to_host", "cuda_to_cuda"])
def memory_type(request):
    return request.param
