import pytest
from efa_common import has_rdma

# The memory types for bi-directional tests.
memory_type_list_bi_dir = [
    pytest.param("host_to_host"),
    pytest.param("host_to_cuda", marks=pytest.mark.cuda_memory),
    pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory),
    pytest.param("host_to_neuron", marks=pytest.mark.neuron_memory),
    pytest.param("neuron_to_neuron", marks=pytest.mark.neuron_memory),
]

# Add more memory types that are useful for uni-directional tests.
memory_type_list_all = memory_type_list_bi_dir + [
    pytest.param("cuda_to_host", marks=pytest.mark.cuda_memory),
    pytest.param("neuron_to_host", marks=pytest.mark.neuron_memory),
]

@pytest.fixture(scope="module", params=memory_type_list_all)
def memory_type(request):
    return request.param

@pytest.fixture(scope="module", params=memory_type_list_bi_dir)
def memory_type_bi_dir(request):
    return request.param

@pytest.fixture(scope="module", params=["read", "writedata", "write"])
def rma_operation_type(request):
    return request.param

@pytest.fixture(scope="module")
def rma_bw_memory_type(memory_type, rma_operation_type):
    is_test_bi_dir = False if rma_operation_type == "writedata" else True
    if is_test_bi_dir and (memory_type not in [_.values[0] for _ in memory_type_list_bi_dir]):
        pytest.skip("Duplicated memory type for bi-directional test")
    return memory_type

@pytest.fixture(scope="function")
def rma_bw_completion_semantic(cmdline_args, completion_semantic, rma_operation_type):
    if completion_semantic != 'delivery_complete':
        # There is no difference between DC and non-DC for read as it's
        # not a transmission
        if rma_operation_type == 'read':
            pytest.skip("Duplicate completion semantic for fi_read test")
        assert rma_operation_type in ['write', 'writedata']
        # If device support rdma write, all the transmissions are DC
        if has_rdma(cmdline_args, 'write'):
            pytest.skip("Duplicate completion semantic for fi_write* test")
    return completion_semantic


@pytest.fixture(scope="module", params=["r:0,4,64",
                                        "r:4048,4,4148",
                                        "r:8000,4,9000",
                                        "r:17000,4,18000",
                                        "r:0,1024,1048576"])
def message_size(request):
    return request.param


@pytest.fixture(scope="module", params=["r:0,4,64",
                                        "r:4048,4,4148",
                                        "r:8000,4,9000",])
def inject_message_size(request):
    return request.param


@pytest.fixture(scope="module", params=["r:0,4,32",
                                        "r:0,1024,8192",])
def zcpy_recv_message_size(request):
    return request.param

@pytest.fixture(scope="module")
def zcpy_recv_max_msg_size(request):
    return 8192

@pytest.fixture(scope="module", params=["r:0,4,32",
                                        "r:0,1024,8192",])
def direct_message_size(request):
    return request.param

# TODO: Include 0 byte test when we support 0 byte rma inject
@pytest.fixture(scope="module", params=["r:1,4,32",
                                        "r:1,1024,8192",])
def direct_rma_size(request):
    return request.param

@pytest.fixture(scope="module", params=["efa", "efa-direct"])
def fabric(request):
    return request.param

@pytest.fixture(scope="function")
def rma_fabric(cmdline_args, fabric):
    if fabric == 'efa-direct' and (
        not has_rdma(cmdline_args, 'read') or
        not has_rdma(cmdline_args, 'write') or
        not has_rdma(cmdline_args, 'writedata')
    ):
        pytest.skip("FI_RMA is not supported. Skip rma tests on efa-direct.")
    return fabric

@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(session, config, items):
    # Called after collection has been performed, may filter or re-order the items in-place
    # We use this hook to always run the MR exhaustion test at the end
    mr_exhaustion_tests, other_tests = [], []
    for item in items:
        if "mr_exhaustion" in item.name:
            mr_exhaustion_tests.append(item)
        else:
            other_tests.append(item)

    yield other_tests + mr_exhaustion_tests
