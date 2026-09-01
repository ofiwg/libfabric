import os
import pytest
import subprocess
import time
from retrying import retry
from common import (
    client_server_have_device,
    num_hmem_devices,
    test_selected_by_marker,
    has_ssh_connection_err_msg,
    is_ssh_connection_error,
    SshConnectionError,
)
from efa_common import (
    has_rdma,
    support_cq_interrupts,
    CudaMemorySupport,
    get_cuda_memory_support,
    get_efa_device_names,
    memory_type_list_bi_dir,
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


# PR CI test types, most specific first. A per test type marker kwarg
# (memory_type(pr_ci_hmem=...), message_sizes(pr_ci_hmem_efa=...)) is looked up
# in this order, so an HMEM run falls back to the plain PR CI declaration when a
# test does not need a different HMEM matrix.
TEST_TYPE_FALLBACKS = {
    "pr_ci_hmem": ("pr_ci_hmem", "pr_ci"),
    "pr_ci": ("pr_ci",),
    "default": ("default",),
}


def get_test_type(test_markers, config):
    """
    Return the test type the current run selected this test with:
    'pr_ci_hmem' for the PR CI suite on an accelerator instance, 'pr_ci' for the
    PR CI suite elsewhere, else 'default'.

    The type is decided by which marker caused the test to be selected, so the
    harness controls it with -t (runfabtests) / -m (pytest) and no hardware
    detection is involved.
    """
    if test_selected_by_marker(config, test_markers, "pr_ci_hmem"):
        return "pr_ci_hmem"
    if test_selected_by_marker(config, test_markers, "pr_ci"):
        return "pr_ci"
    return "default"


def marker_kwarg_for_test_type(marker, test_type, suffix=""):
    """
    Return (kwarg_name, value) for the most specific test type a marker declares,
    or (None, None). suffix lets the message_sizes marker key on fabric too, e.g.
    test_type 'pr_ci_hmem' + suffix '_efa' looks up 'pr_ci_hmem_efa' then
    'pr_ci_efa'.
    """
    for candidate in TEST_TYPE_FALLBACKS.get(test_type, (test_type,)):
        name = candidate + suffix
        if name in marker.kwargs:
            return name, marker.kwargs[name]
    return None, None


FABRIC_KWARG_SUFFIX = {"efa": "_efa", "efa-direct": "_efa_direct"}


def choose_message_sizes_for_fabric_test_type(fabric, test_type, sizes_marker, nodeid):
    """
    Return the message-size list for (fabric, test_type) from a
    @pytest.mark.message_sizes marker.
    example:
    @pytest.mark.message_sizes(default_efa=PERF_SIZES, pr_ci_efa=DIRECT_SIZES)
                   ^sizes marker   ^kwarg_name   ^kwarg_sizes

    A kwarg name is <test_type>_<fabric>, matched exactly, walking the test type
    fallback chain: an HMEM PR CI run uses the pr_ci_* sizes unless the test
    declares pr_ci_hmem_* ones. Matching the whole name matters because
    'pr_ci' is a prefix of 'pr_ci_hmem'.
    """
    suffix = FABRIC_KWARG_SUFFIX.get(fabric)
    if suffix is None:
        raise ValueError(f"Unknown fabric: {fabric!r}")

    _, sizes = marker_kwarg_for_test_type(sizes_marker, test_type, suffix)
    if not sizes:
        raise ValueError(
            f"@pytest.mark.message_sizes on {nodeid} is missing a kwarg for "
            f"fabric={fabric!r} test_type={test_type!r} "
            f"(have {sorted(sizes_marker.kwargs)})"
        )
    return sizes


def kwarg_test_type(kwarg_name):
    """
    Return the test type part of a message_sizes kwarg name, i.e. the name with
    its trailing _<fabric> stripped: 'pr_ci_hmem_efa_direct' -> 'pr_ci_hmem'.
    """
    for suffix in sorted(FABRIC_KWARG_SUFFIX.values(), key=len, reverse=True):
        if kwarg_name.endswith(suffix):
            return kwarg_name[: -len(suffix)]
    return kwarg_name


def choose_message_sizes_for_test_type(test_type, sizes_marker, nodeid):
    """
    Return the message-size list for test_type from a @pytest.mark.message_sizes
    marker on a test with no fabric parametrization.

    The test type part of each kwarg name is compared exactly, walking the
    fallback chain, so an HMEM PR CI run uses the pr_ci_* sizes unless the test
    declares pr_ci_hmem_* ones, and 'pr_ci' never matches a 'pr_ci_hmem_*'
    kwarg by prefix.
    """
    for candidate in TEST_TYPE_FALLBACKS.get(test_type, (test_type,)):
        sizes = []
        for kwarg_name, kwarg_sizes in sizes_marker.kwargs.items():
            if kwarg_test_type(kwarg_name) == candidate:
                sizes.extend(kwarg_sizes)
        if sizes:
            return sizes
    raise ValueError(
        f"@pytest.mark.message_sizes on {nodeid} has "
        f"no kwarg naming {test_type!r} (have {sorted(sizes_marker.kwargs)})"
    )


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
    metafunc.parametrize("message_sizes",
                         choose_message_sizes_for_test_type(test_type, sizes_marker,
                                                            metafunc.definition.nodeid))


def pick_preferred_memory_flavor(candidates, detected):
    """
    Flavor selection for @pytest.mark.memory_type(..., prefer_accelerator=True):
    pick one memory flavor at test generation time to reduce the test count.

    Returns the detected accelerator memory types when the endpoints have an
    accelerator, host memory otherwise.  If detection fails, return everything.

    Anything other than host_to_host counts as an accelerator flavor. With a
    candidate list containing mixed flavors (host_to_cuda, cuda_to_host, ...),
    every detected mixed flavor is kept alongside the device_to_device one.
    """
    host_only = [p for p in candidates if p.values[0] == "host_to_host"]

    if detected is None:
        # detection unavailable: run every candidate
        return candidates

    accelerator = [p for p in detected if p.values[0] != "host_to_host"]
    return accelerator or host_only


def add_memory_type_parametrization(metafunc, memory_type_marker, test_type):
    """
    Parametrize the memory_type fixture at collection time from the test's
    @pytest.mark.memory_type(...) declaration, dropping any permutation whose
    device is absent on the owning endpoint.

    The candidate list is the one the marker declares for the running test type,
    most specific first:

        @pytest.mark.memory_type(memory_type_list_all,             # default runs
                                 pr_ci=memory_type_list_symm,      # PR CI
                                 pr_ci_hmem=memory_type_list_all)  # PR CI, GPU

    A missing kwarg falls back to the less specific one and finally to the
    positional list, so forgetting a kwarg widens coverage rather than silently
    dropping it.

    Fallback (no coverage regression): if --server-id/--client-id are not
    provided or device detection fails, every candidate memory type is
    included and the runtime skip in common.py remains the safety net.

    prefer_accelerator=True picks one memory flavor at generation time to
    reduce the test count; see pick_preferred_memory_flavor().
    """

    if "memory_type" not in metafunc.fixturenames:
        return

    # A test consuming the memory_type fixture must declare a memory_type marker
    # whose argument is a memory_type_list_* from efa_common.
    if memory_type_marker is None:
        raise ValueError(
            f"{metafunc.definition.nodeid} consumes the memory_type fixture "
            f"but is missing @pytest.mark.memory_type(...)"
        )

    prefer_accelerator = memory_type_marker.kwargs.get("prefer_accelerator", False)

    _, candidates = marker_kwarg_for_test_type(memory_type_marker, test_type)
    if candidates is None:
        candidates = memory_type_marker.args[0]

    server_id = metafunc.config.getoption("--server-id", default=None)
    client_id = metafunc.config.getoption("--client-id", default=None)

    # detected is None when device detection could not run
    detected = None
    if server_id and client_id:
        try:
            detected = [
                param for param in candidates
                if client_server_have_device(param.values[0], server_id, client_id)
            ]
        except Exception:
            # Fallback when detection/SSH fails
            detected = None

    if prefer_accelerator:
        params = pick_preferred_memory_flavor(candidates, detected)
    else:
        params = candidates if detected is None else detected

    metafunc.parametrize("memory_type", params, scope="module")


def pytest_generate_tests(metafunc):
    """
    Derive parametrization from markers
      - @pytest.mark.pr_ci
      - @pytest.mark.fabric(params=[...])
      - @pytest.mark.message_sizes(<test_type>_<fabric>=..., ...)
      - @pytest.mark.memory_type(memory_type_list_*)
    the last also filtering by endpoint device availability.
    """
    # get all markers
    fabric_marker = next(metafunc.definition.iter_markers("fabric"), None)
    sizes_marker  = next(metafunc.definition.iter_markers("message_sizes"), None)
    memory_type_marker = next(metafunc.definition.iter_markers("memory_type"), None)

    # find out the test type running from markers (currently pr_ci or default)
    test_markers = {m.name for m in metafunc.definition.iter_markers()}
    test_type = get_test_type(test_markers, metafunc.config)

    # generate parametrization based on found markers and test type
    add_fabric_and_message_size_parametrization(metafunc, fabric_marker, sizes_marker, test_type)

    # parametrize memory_type from its marker, dropping permutations
    # whose device is absent on the owning endpoint
    add_memory_type_parametrization(metafunc, memory_type_marker, test_type)

hmem_type_list = [
    pytest.param("cuda", marks=pytest.mark.cuda_memory),
    pytest.param("neuron", marks=pytest.mark.neuron_memory),
]

@pytest.fixture(scope="module", params=hmem_type_list)
def hmem_type(request):
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


@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def device_has_hw_cntr(host_id):
    """
    Return True if the EFA device reports a non-zero hardware
    counter count, indiciating hardware counter support
    """
    command = "ssh {} 'fi_info -p efa -v || /opt/amazon/efa/bin/fi_info -p efa -v' | grep cntr_cnt".format(host_id)
    proc = subprocess.run(command, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", timeout=60)

    if has_ssh_connection_err_msg(proc.stdout) or has_ssh_connection_err_msg(proc.stderr):
        raise SshConnectionError()

    for line in proc.stdout.strip().split("\n"):
        value = line.split(":")[-1].strip()
        if value.isdigit() and int(value) > 0:
            return True
    return False


def pytest_collection_modifyitems(session, config, items):
    # Called after collection has been performed, deselects tests whose
    # required binary or device support is missing. Test ordering is handled
    # by the shared hook in the parent conftest.
    binpath = config.getoption("--binpath", default="") or ""
    server_id = config.getoption("--server-id", default="")
    client_id = config.getoption("--client-id", default="")
    have_hw_cntr = os.path.exists(os.path.join(binpath, "fi_efa_hw_cntr"))
    if have_hw_cntr:
        for host_id in filter(None, (server_id, client_id)):
            try:
                if not device_has_hw_cntr(host_id):
                    have_hw_cntr = False
                    break
            except SshConnectionError:
                pytest.fail(
                    "Could not determine hw_cntr support: ssh to {} failed after "
                    "retries. Refusing to silently deselect hw_cntr tests.".format(host_id))
    have_gda = os.path.exists(os.path.join(binpath, "fi_efa_gda"))

    deselected = []
    remaining = []
    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "hw_cntr" in markers and not have_hw_cntr:
            deselected.append(item)
        elif "gda" in markers and not have_gda:
            deselected.append(item)
        else:
            remaining.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


@pytest.fixture(scope="function")
def support_sread(cmdline_args):
    """Check if both server and client support cq interrupts."""
    return (support_cq_interrupts(cmdline_args.server_id) and
            support_cq_interrupts(cmdline_args.client_id))


@pytest.fixture(scope="session")
def num_domains(request):
    """
    Number of EFA domains (NICs) a test can spread endpoints across.

    This is the smaller of the two hosts' device counts: each peer opens
    domains from its own device list, so a heterogeneous pair can only spread
    as far as the host with fewer devices.

    Session-scoped so the device lookup runs once per xdist worker instead of
    once per test. The server/client ids are read from the session scoped request
    fixture
    """
    server_id = request.config.getoption("--server-id")
    client_id = request.config.getoption("--client-id")

    counts = [len(get_efa_device_names(host_id))
              for host_id in dict.fromkeys(filter(None, (server_id, client_id)))]

    if not counts or not min(counts):
        pytest.skip("could not determine the EFA device count on both hosts")

    return min(counts)
