import pytest

@pytest.fixture(scope="module", params=["host_to_host", "host_to_cuda",
                                        "cuda_to_host", "cuda_to_cuda"])
def memory_type(request):
    return request.param

@pytest.fixture(scope="module", params=["r:8000,4,9000", 
                                         "r:0,4,64", "r:4048,4,4148"])
def message_size(request):
    return request.param
