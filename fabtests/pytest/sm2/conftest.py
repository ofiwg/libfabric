import pytest

# TODO Restore GPU Memory Types
@pytest.fixture(scope="module", params=["host_to_host"])
def memory_type(request):
    return request.param
