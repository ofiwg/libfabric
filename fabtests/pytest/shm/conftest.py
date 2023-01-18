import pytest


@pytest.fixture(scope="module", params=["host_to_host",
                                        pytest.param("host_to_cuda", marks=pytest.mark.cuda_memory),
                                        pytest.param("cuda_to_host", marks=pytest.mark.cuda_memory),
                                        pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory),
                                        pytest.param("neuron_to_neuron", marks=pytest.mark.neuron_memory),
                                        pytest.param("neuron_to_host", marks=pytest.mark.neuron_memory),
                                        pytest.param("host_to_neuron", marks=pytest.mark.neuron_memory)])
def memory_type(request):
    return request.param