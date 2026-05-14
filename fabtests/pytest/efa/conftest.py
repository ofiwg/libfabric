import pytest
import time
from common import test_selected_by_marker
from efa_common import (
    has_rdma, 
    support_cq_interrupts,
    CudaMemorySupport,
    get_cuda_memory_support,
)

# Message size lists are defined in efa_common.py and imported by test files directly.
# The pytest_generate_tests hook reads them from the @pytest.mark.message_sizes decorator.


def fabric_present(string, fabric):
    """
    Return true if 'fabric' is present in 'string', raises error if the
    fabric is invalid
    """
    if fabric == "efa":
        return "efa" in string and "efa_direct" not in string
    if fabric == "efa-direct":
        return "efa_direct" in string
    raise ValueError(f"Unknown fabric: {fabric!r}")


def get_test_type(test_markers, config):
    """
    Return 'pr_ci' if this test is being collected because of the pr_ci marker,
    else 'default'.
    """
    if test_selected_by_marker(config, test_markers, "pr_ci"):
        return "pr_ci"
    return "default"


def choose_message_sizes_for_fabric_test_type(fabric, test_type, sizes_marker, nodeid):
    """
    Return all matching message-size lists for (fabric, test_type) from a
    @pytest.mark.message_sizes marker.
    example:
    @pytest.mark.message_sizes(default_efa=PERF_SIZES, pr_ci_efa=DIRECT_RMA_SIZES)
                   ^sizes marker   ^kwarg_name   ^kwarg_sizes
    """
    sizes = []
    for kwarg_name, kwarg_sizes in sizes_marker.kwargs.items():
        # add message sizes if they match both the fabric and the test type
        if fabric_present(kwarg_name, fabric) and test_type in kwarg_name:
            sizes.extend(kwarg_sizes)
    if not sizes:
        raise ValueError(
            f"@pytest.mark.message_sizes on {nodeid} is missing a kwarg for "
            f"fabric={fabric!r} test_type={test_type!r} "
            f"(have {sorted(sizes_marker.kwargs)})"
        )
    return sizes


def add_fabric_and_message_size_parametrization(metafunc, fabric_marker, sizes_marker, test_type):
    # look at markers and find out if this test specifies fabric and message sizes
    wants_fabric = fabric_marker is not None and "fabric" in metafunc.fixturenames
    wants_sizes  = sizes_marker  is not None and "message_sizes" in metafunc.fixturenames

    # no parametrization needed
    if not wants_fabric and not wants_sizes:
        return

    # get message size based on fabric and test type
    if wants_fabric and wants_sizes:
        nodeid = metafunc.definition.nodeid
        params = []
        for fabric in fabric_marker.kwargs["params"]:
            for size in choose_message_sizes_for_fabric_test_type(fabric, test_type, sizes_marker, nodeid):
                params.append(pytest.param(fabric, size))
        metafunc.parametrize(("fabric", "message_sizes"), params, indirect=["fabric"])
        return

    # no message size param, just add fabric parametrization
    if wants_fabric:
        metafunc.parametrize("fabric", fabric_marker.kwargs["params"], indirect=True)
        return

    # no fabric param, just add message sizes parametrization based on test type
    sizes = []
    for k, kwarg_sizes in sizes_marker.kwargs.items():
        if test_type in k:
            sizes.extend(kwarg_sizes)
    if not sizes:
        raise ValueError(
            f"@pytest.mark.message_sizes on {metafunc.definition.nodeid} has "
            f"no kwarg naming {test_type!r} (have {sorted(sizes_marker.kwargs)})"
        )
    metafunc.parametrize("message_sizes", sizes)


def pytest_generate_tests(metafunc):
    """
    Derive parametrization from markers
      - @pytest.mark.pr_ci
      - @pytest.mark.fabric(params=[...])
      - @pytest.mark.message_sizes(<test_type>_<fabric>=..., ...)
    """
    # get all markers
    fabric_marker = next(metafunc.definition.iter_markers("fabric"), None)
    sizes_marker  = next(metafunc.definition.iter_markers("message_sizes"), None)

    # find out the test type running from markers (currently pr_ci or default)
    test_markers = {m.name for m in metafunc.definition.iter_markers()}
    test_type = get_test_type(test_markers, metafunc.config)

    # generate parametrization based on found markers and test type
    add_fabric_and_message_size_parametrization(metafunc, fabric_marker, sizes_marker, test_type)

# The memory types for bi-directional tests.
memory_type_list_bi_dir = [
    pytest.param("host_to_host"),
    pytest.param("host_to_cuda", marks=pytest.mark.cuda_memory),
    pytest.param("cuda_to_cuda", marks=pytest.mark.cuda_memory),
    pytest.param("host_to_neuron", marks=pytest.mark.neuron_memory),
    pytest.param("neuron_to_neuron", marks=pytest.mark.neuron_memory),
    pytest.param("host_to_rocr", marks=pytest.mark.rocr_memory),
    pytest.param("rocr_to_rocr", marks=pytest.mark.rocr_memory),
]

# Add more memory types that are useful for uni-directional tests.
memory_type_list_all = memory_type_list_bi_dir + [
    pytest.param("cuda_to_host", marks=pytest.mark.cuda_memory),
    pytest.param("neuron_to_host", marks=pytest.mark.neuron_memory),
    pytest.param("rocr_to_host", marks=pytest.mark.rocr_memory),
]

hmem_type_list = [
    pytest.param("cuda", marks=pytest.mark.cuda_memory),
    pytest.param("neuron", marks=pytest.mark.neuron_memory),
]

@pytest.fixture(scope="module", params=hmem_type_list)
def hmem_type(request):
    return request.param

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


@pytest.fixture(scope="function")
def fabric(request):
    return request.param


@pytest.fixture(scope="function")
def rma_fabric(cmdline_args, fabric):
    if fabric == "efa-direct" and (
        not has_rdma(cmdline_args, "read")
        or not has_rdma(cmdline_args, "write")
        or not has_rdma(cmdline_args, "writedata")
    ):
        pytest.skip("FI_RMA is not supported. Skip rma tests on efa-direct.")
    return fabric


@pytest.fixture(scope="function", params=["rx-cq-data", "no-rx-cq-data"])
def rx_cq_data_cli(request, fabric, rma_operation_type):
    if request.param == "no-rx-cq-data":
        if rma_operation_type != "writedata":
            pytest.skip("the rx cq data mode is only applied for writedata")
        if fabric == "efa-direct" :
            return " --no-rx-cq-data"
        else:
            pytest.skip("efa fabric ignores the rx cq data mode")
    return " "


def cuda_memory_type_validation(cmdline_args):
    """
    Validate CUDA memory type configuration against hardware capabilities at session startup.
    
    Args:
        cmdline_args: Command line arguments containing dmabuf configuration.
        
    Returns:
        None
        
    Notes:
        - Skips tests if user specified non-dmabuf but hardware only supports DMA_BUF_ONLY
        - Only validates if CUDA tests are being run
    """
    # Check if CUDA tests are being run via expression
    print("Running cuda_memory_type_validation() validation checks!")
    
    cuda_support: CudaMemorySupport = get_cuda_memory_support(
                                            cmdline_args=cmdline_args, 
                                            ip=cmdline_args.server_id
                                        )

    if cuda_support == CudaMemorySupport.NOT_INITIALIZED:
        pytest.fail("CUDA memory support never initialized")
    
    do_dmabuf = cmdline_args.do_dmabuf_reg_for_hmem
    
    print(f"Correctly defined dma buf mode {do_dmabuf} and return {cuda_support}!")
    
    return


@pytest.fixture(scope="function", autouse=True)
def cuda_validation_fixture(request, cmdline_args):
    """Auto-run CUDA validation if CUDA tests are present."""
    # Check if the current test has cuda_memory mark
    has_cuda_mark = any(mark.name == 'cuda_memory' for mark in request.node.iter_markers())
    
    if has_cuda_mark:
        cuda_memory_type_validation(cmdline_args)
    else:
        print("No CUDA memory mark, skipping validation")


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


@pytest.fixture(scope="function")
def support_sread(cmdline_args):
    """Check if both server and client support cq interrupts."""
    return (support_cq_interrupts(cmdline_args.server_id) and
            support_cq_interrupts(cmdline_args.client_id))


