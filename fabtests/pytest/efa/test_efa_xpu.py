"""
fi_efa_xpu: FI_XPU on efa-direct, a kernel posting to and reading from the
NIC on the GPU (see prov/efa/src/efa_xpu/efa_xpu.c).

A run is one launch shape - an operation, a scope, how many threads and
blocks - and one way of completing its operations: the device reads the CQ
(through one poller, or with the issuers reading it themselves, --readers),
the host reads it (--host-cq), or counters (--use-cntr). Every test also runs
with the send queue's 64-bit request IDs turned off, which sends the send
queue through the context table that stands in for them on hardware without
them. Verified runs (-v) check every byte each issuer sends.

The runs are independent, so they suit xdist: each worker gets its own GPU,
the EFA device nearest it and its own OOB port (see ClientServerTest), which
needs runfabtests.py -b.
"""

import copy

import pytest
from common import ClientServerTest

pytestmark = [
    pytest.mark.functional,
    pytest.mark.cuda_memory,
    pytest.mark.xpu,
    pytest.mark.fabric(params=["efa-direct"]),
]

OPS = ["msg", "write", "writedata", "read"]


def _shape(args, id):
    return pytest.param(args, id=id)


# One operation per issuer, and what makes an issuer is the scope: a thread,
# a warp or a block. The shapes cover a warp that ends short, and several
# blocks posting into one queue.
SHAPES = [
    _shape("--scope work_item --threads 7", "work_item-7"),
    _shape("--scope work_item --threads 256 -W 256", "work_item-256"),
    _shape("--scope work_item --threads 64 --blocks 4 -W 256", "work_item-64x4"),
    _shape("--scope subgroup --threads 32", "subgroup-32"),
    _shape("--scope subgroup --threads 33", "subgroup-33"),
    _shape("--scope subgroup --threads 200 --blocks 4", "subgroup-200x4"),
    _shape("--scope work_group --threads 64", "work_group-64"),
    _shape("--scope work_group --threads 1024 --blocks 8", "work_group-1024x8"),
]

ONE_PER_SCOPE = [
    _shape("--scope work_item --threads 256 -W 256", "work_item-256"),
    _shape("--scope subgroup --threads 256", "subgroup-256"),
    _shape("--scope work_group --threads 256 --blocks 4", "work_group-256x4"),
]


@pytest.fixture(params=["req_id_64", "ctx_table"])
def xpu_cmdline_args(request, cmdline_args):
    """The send queue's 64-bit request IDs as the device offers them, or
    turned off, so the send queue runs through a context table."""
    if request.param == "req_id_64":
        return cmdline_args
    args = copy.copy(cmdline_args)
    args.append_environ("FI_EFA_USE_SQ_REQ_ID_64_BIT=0")
    return args


def run_xpu(cmdline_args, fabric, args):
    test = ClientServerTest(cmdline_args, "fi_efa_xpu " + args,
                            memory_type="cuda_to_cuda", fabric=fabric)
    test.run()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("op", OPS)
def test_xpu_bw(xpu_cmdline_args, fabric, op, shape):
    """Every operation at every scope, verified, the device reading the CQ."""
    run_xpu(xpu_cmdline_args, fabric, f"-v -o {op} {shape}")


@pytest.mark.parametrize("op,shape", [
    pytest.param("msg", "--scope work_item --threads 64", id="msg-work_item-64"),
    pytest.param("msg", "--scope subgroup --threads 64", id="msg-subgroup-64"),
    pytest.param("write", "--scope work_group --threads 64 --blocks 2",
                 id="write-work_group-64x2"),
    pytest.param("read", "--scope subgroup --threads 33 --blocks 2",
                 id="read-subgroup-33x2"),
])
def test_xpu_bw_cntr(xpu_cmdline_args, fabric, op, shape):
    """Completions counted on counters; the CQ is still drained."""
    run_xpu(xpu_cmdline_args, fabric, f"-v -o {op} {shape} --use-cntr")


@pytest.mark.parametrize("shape", ONE_PER_SCOPE)
@pytest.mark.parametrize("op", ["msg", "write"])
def test_xpu_bw_perf(xpu_cmdline_args, fabric, op, shape):
    """Bandwidth over every message size, unverified."""
    run_xpu(xpu_cmdline_args, fabric, f"-o {op} {shape}")


@pytest.mark.parametrize("verify", [pytest.param("-v", id="verified"),
                                    pytest.param("", id="unverified")])
@pytest.mark.parametrize("shape", [
    _shape("--scope work_item --threads 1", "work_item-1"),
    _shape("--scope work_item --threads 32", "work_item-32"),
    _shape("--scope subgroup --threads 64", "subgroup-64"),
    _shape("--scope work_group --threads 64 --blocks 2", "work_group-64x2"),
])
def test_xpu_lat(xpu_cmdline_args, fabric, shape, verify):
    """Ping-pong latency, one operation outstanding at a time."""
    run_xpu(xpu_cmdline_args, fabric, f"{verify} -o msg --lat {shape}")


@pytest.mark.parametrize("shape", [
    _shape("--scope work_item --threads 1", "work_item-1"),
    _shape("--scope work_item --threads 64", "work_item-64"),
    _shape("--scope subgroup --threads 128", "subgroup-128"),
    _shape("--scope work_group --threads 256 --blocks 4", "work_group-256x4"),
])
@pytest.mark.parametrize("op", OPS)
def test_xpu_host_cq(xpu_cmdline_args, fabric, op, shape):
    """The kernel only posts, many times the queue's depth, and the host reads
    the CQ: a post waits for room rather than overrunning the queue."""
    run_xpu(xpu_cmdline_args, fabric, f"--host-cq -o {op} {shape}")


@pytest.mark.parametrize("shape", [
    _shape("--scope work_item --threads 32 --eps 4", "work_item-32-eps4"),
    _shape("--scope work_item --threads 64 --blocks 2 --eps 3 -W 256",
           "work_item-64x2-eps3"),
])
@pytest.mark.parametrize("op", OPS)
def test_xpu_multi_qp(xpu_cmdline_args, fabric, op, shape):
    """A QP per lane: issuer i posts on endpoint i % N to the peer's endpoint
    i % N, so adjacent work item lanes post to different QPs."""
    run_xpu(xpu_cmdline_args, fabric, f"-v -o {op} {shape}")


@pytest.mark.parametrize("args", [
    pytest.param("-v -o msg --scope subgroup --threads 128 --eps 2",
                 id="subgroup-eps2"),
    pytest.param("-v -o write --scope work_group --threads 64 --blocks 4 --eps 4",
                 id="work_group-eps4"),
    pytest.param("-v -o msg --scope work_item --threads 64 --eps 4 --use-cntr",
                 id="cntr-eps4"),
    pytest.param("-v -o msg --lat --scope work_item --threads 32 --eps 4",
                 id="lat-eps4"),
    pytest.param("-v -o write --scope work_item --threads 64 --eps 8",
                 id="work_item-eps8"),
    pytest.param("-v -o msg --scope work_item --threads 128 --eps 16 -W 256",
                 id="work_item-eps16"),
    pytest.param("--host-cq -o write --scope work_item --threads 64 --eps 4",
                 id="host_cq-write-eps4"),
    pytest.param("--host-cq -o msg --scope work_item --threads 64 --eps 16",
                 id="host_cq-msg-eps16"),
])
def test_xpu_multi_qp_modes(xpu_cmdline_args, fabric, args):
    """A QP per lane with the other completion readers and more endpoints,
    which split each endpoint's queue so the shared CQ fits."""
    run_xpu(xpu_cmdline_args, fabric, args)


@pytest.mark.parametrize("op", OPS)
def test_xpu_readers(xpu_cmdline_args, fabric, op):
    """The issuers read the CQ themselves, many at once: overlapping reads,
    each taking completions of its own."""
    run_xpu(xpu_cmdline_args, fabric,
            f"-v -o {op} --scope work_item --threads 64 --readers")


@pytest.mark.parametrize("args", [
    pytest.param("-v -o msg --scope work_item --threads 64 --blocks 4 -W 256",
                 id="work_item-64x4"),
    pytest.param("-v -o write --scope subgroup --threads 128", id="subgroup-128"),
    pytest.param("-v -o msg --scope work_group --threads 64 --blocks 4",
                 id="work_group-64x4"),
    pytest.param("-v -o msg --scope work_item --threads 64 --eps 4", id="eps4"),
    pytest.param("-v -o msg --lat --scope work_item --threads 32", id="lat"),
])
def test_xpu_readers_modes(xpu_cmdline_args, fabric, args):
    """Overlapping reads across blocks, at the group scopes, with a QP per
    lane and in a ping-pong."""
    run_xpu(xpu_cmdline_args, fabric, f"{args} --readers")


def test_xpu_unknown_scope_refused(cmdline_args, fabric):
    """A scope the header does not know is refused; the run also checks on the
    GPU that work item lanes are batched by endpoint."""
    run_xpu(cmdline_args, fabric, "-o msg --scope unknown")
