import pytest
from common import ClientServerTest
from efa.efa_common import memory_type_list_symm


pytestmark = pytest.mark.pre_release


# fi_mr_abort allocates -W MRs and posts -N transfers per MR, so a run
# submits up to W*N operations per iteration; more in-flight operations
# means more NIC load and memory pressure. Every test opens
# MR_ABORT_NUM_EPS endpoints per side, packed MR_ABORT_EPS_PER_DOMAIN to
# a domain, and MRs (and thus their ops) are distributed round-robin
# across the initiator endpoints. Cap the load per endpoint at
# MR_ABORT_OPS_PER_EP by deriving -W from -N and the endpoint count, so
# every domain carries MR_ABORT_EPS_PER_DOMAIN * MR_ABORT_OPS_PER_EP
# ops per iteration regardless of how many domains a test spreads over.
MR_ABORT_NUM_EPS = 4
MR_ABORT_EPS_PER_DOMAIN = 4
MR_ABORT_OPS_PER_EP = 512

MR_ABORT_EP_ARGS = (f" --num-eps {MR_ABORT_NUM_EPS}"
                    f" --eps-per-domain {MR_ABORT_EPS_PER_DOMAIN}")


def mr_abort_num_mrs(ops_per_mr, num_eps=MR_ABORT_NUM_EPS):
    """-W value that keeps the load at MR_ABORT_OPS_PER_EP ops per endpoint."""
    return num_eps * MR_ABORT_OPS_PER_EP // ops_per_mr

MSG_SIZE_64B = 64
MSG_SIZE_128B = 128
MSG_SIZE_4KIB = 4096
MSG_SIZE_8KIB = 8192
MSG_SIZE_64KIB = 65536
MSG_SIZE_128KIB = 131072
MSG_SIZE_256KIB = 262144
MSG_SIZE_1MIB = 1048576
MSG_SIZE_10MIB = 10485760

BASE_MESSAGE_SIZES = [
    MSG_SIZE_64B,
    # 128B is the largest write WQE the low latency service level
    # accelerates; only low-latency cases are generated at this size
    MSG_SIZE_128B,
    MSG_SIZE_4KIB,
    MSG_SIZE_64KIB,
    MSG_SIZE_1MIB,
    MSG_SIZE_10MIB,
]

# The high PPS WQE hint only accelerates writes <= 8KB, but writes up to
# 64KB marked high-pps exercise special handling that routes them back to
# the default processing path. Request high PPS for writes up to 64KB to
# utilize both paths; larger sizes add no new coverage.
HIGH_PPS_MAX_WRITE_SIZE = MSG_SIZE_64KIB

# The low latency service level only accelerates write WQEs <= 128 bytes,
# but request it for writes up to 8 KiB so a single run mixes the
# accelerated (<= 128B) and non-accelerated flows.
SL_LOW_LATENCY_MAX_WRITE_SIZE = MSG_SIZE_8KIB


def rma_case_params():
    params = []
    for rma_op in ["write", "read", "writedata"]:
        for size in BASE_MESSAGE_SIZES:

            use_high_pps_vals = [None]
            use_sl_low_latency_vals = [None]

            if (rma_op == "write" or rma_op == "writedata"):
                # High PPS parametrization is only applicable for RDMA writes
                # up to 64KB
                if size <= HIGH_PPS_MAX_WRITE_SIZE:
                    use_high_pps_vals = [True, False]

                # Low latency SL is only applicable for RDMA writes up to 8 KiB
                if size <= SL_LOW_LATENCY_MAX_WRITE_SIZE:
                    use_sl_low_latency_vals = [True, False]

            marks = []
            if size == MSG_SIZE_10MIB:
                # 10 MiB: large transfers consume far more memory/NIC resources
                # Run this size serially to avoid resource contention with
                # parallel workers
                marks = [pytest.mark.serial]

            for use_high_pps in use_high_pps_vals:
                for use_sl_low_latency in use_sl_low_latency_vals:
                    # high_pps and sl_low_latency are mutually exclusive:
                    # do not generate test cases with both enabled
                    if use_high_pps and use_sl_low_latency:
                        continue

                    # 128B exists only to pin the low latency SL firmware
                    # boundary; skip every other combination at that size
                    if size == MSG_SIZE_128B and not use_sl_low_latency:
                        continue

                    id = f"{rma_op}-{size}"

                    if use_high_pps is not None:
                        id += f"-high_pps-{use_high_pps}"

                    if use_sl_low_latency is not None:
                        id += f"-sl_low_lat-{use_sl_low_latency}"

                    params.append(pytest.param(size, rma_op, use_high_pps,
                                               use_sl_low_latency, marks=marks,
                                               id=id))
    return params


def abort_case_params():
    """
    Expand rma_case_params() with the (cancel_order, ops_per_mr)
    combinations that actually run for each message size.

    The 10 MiB size only runs with reverse cancel order on 1 operation per
    MR to save run time; every other size runs the full cross product.
    """
    params = []
    for p in rma_case_params():
        size = p.values[0]
        if size == MSG_SIZE_10MIB:
            cases = [("reverse", 1)]
        else:
            cases = [(cancel_order, ops_per_mr)
                     for cancel_order in ("reverse", "random")
                     for ops_per_mr in (1, 4)]
        for cancel_order, ops_per_mr in cases:
            params.append(pytest.param(*p.values, cancel_order, ops_per_mr,
                                       marks=p.marks,
                                       id=f"{p.id}-{cancel_order}-ops_{ops_per_mr}"))
    return params


# --- Test: abort (RMA) ---
def run_mr_abort(cmdline_args, rma_fabric, rma_op, cancel_order, close_side,
                 ops_per_mr, high_pps, sl_low_latency, message_size,
                 memory_type):
    command = (f"fi_mr_abort -T abort -o {rma_op} -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W {mr_abort_num_mrs(ops_per_mr)}"
               f" -S {message_size}{MR_ABORT_EP_ARGS}")
    if high_pps:
        assert(rma_op != "read")
        command += " --high-pps"

    if sl_low_latency:
        assert(rma_op != "read")
        assert(message_size <= SL_LOW_LATENCY_MAX_WRITE_SIZE)
        command += " --sl-low-latency"

    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=rma_fabric, memory_type=memory_type)
    test.run()


# efa-direct runs one preferred memory flavor; efa runs every detected
# flavor (host_to_host plus the accelerator flavor) so both the system and
# hmem paths are covered.
@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
@pytest.mark.memory_type(memory_type_list_symm, prefer_accelerator=True)
@pytest.mark.parametrize(
    "message_size, rma_op, high_pps, sl_low_latency, cancel_order, ops_per_mr",
    abort_case_params())
def test_mr_abort_efa_direct(cmdline_args, rma_fabric, rma_op, cancel_order,
                             close_side, ops_per_mr, high_pps, sl_low_latency,
                             message_size, memory_type):
    run_mr_abort(cmdline_args, rma_fabric, rma_op, cancel_order, close_side,
                 ops_per_mr, high_pps, sl_low_latency, message_size,
                 memory_type)


@pytest.mark.functional
@pytest.mark.fabric(params=["efa"])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
@pytest.mark.memory_type(memory_type_list_symm)
@pytest.mark.parametrize(
    "message_size, rma_op, high_pps, sl_low_latency, cancel_order, ops_per_mr",
    abort_case_params())
def test_mr_abort_efa(cmdline_args, rma_fabric, rma_op, cancel_order,
                      close_side, ops_per_mr, high_pps, sl_low_latency,
                      message_size, memory_type):
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")
    run_mr_abort(cmdline_args, rma_fabric, rma_op, cancel_order, close_side,
                 ops_per_mr, high_pps, sl_low_latency, message_size,
                 memory_type)


# --- Test: partial (2 MRs on same buffer) ---
def run_mr_abort_partial(cmdline_args, rma_fabric, rma_op, high_pps,
                         sl_low_latency, message_size, memory_type,
                         split_eps):
    command = (f"fi_mr_abort -T partial -o {rma_op} -S {message_size}"
               f"{MR_ABORT_EP_ARGS}")

    if split_eps:
        # Post each slot's surviving and to-be-canceled ops from two
        # different initiator endpoints, so an MR close must not disturb
        # in-flight ops on another QP.
        if MR_ABORT_NUM_EPS < 2:
            # No second endpoint to split across; fi_mr_abort rejects it.
            pytest.skip("--partial-split-eps requires at least 2 "
                        "initiator endpoints")
        if MR_ABORT_EPS_PER_DOMAIN < MR_ABORT_NUM_EPS:
            # With more than one domain the wrap-around split pair
            # (last endpoint -> endpoint 0) always crosses domains, so
            # the test would exercise cross-NIC rather than
            # cross-QP-same-NIC behavior.
            pytest.skip("--partial-split-eps needs all endpoints on one "
                        "domain so every split op pair shares a domain")
        command += " --partial-split-eps"

    if high_pps:
        assert(rma_op != "read")
        command += " --high-pps"

    if sl_low_latency:
        assert(rma_op != "read")
        assert(message_size <= SL_LOW_LATENCY_MAX_WRITE_SIZE)
        command += " --sl-low-latency"

    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=rma_fabric, memory_type=memory_type)
    test.run()


@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.memory_type(memory_type_list_symm, prefer_accelerator=True)
@pytest.mark.parametrize("split_eps", [False, True],
                         ids=["same_ep", "split_eps"])
@pytest.mark.parametrize("message_size, rma_op, high_pps, sl_low_latency",
                         rma_case_params())
def test_mr_abort_partial_efa_direct(cmdline_args, rma_fabric, rma_op,
                                     high_pps, sl_low_latency, message_size,
                                     memory_type, split_eps):
    run_mr_abort_partial(cmdline_args, rma_fabric, rma_op, high_pps,
                         sl_low_latency, message_size, memory_type, split_eps)


@pytest.mark.functional
@pytest.mark.fabric(params=["efa"])
@pytest.mark.memory_type(memory_type_list_symm)
@pytest.mark.parametrize("split_eps", [False, True],
                         ids=["same_ep", "split_eps"])
@pytest.mark.parametrize("message_size, rma_op, high_pps, sl_low_latency",
                         rma_case_params())
def test_mr_abort_partial_efa(cmdline_args, rma_fabric, rma_op, high_pps,
                              sl_low_latency, message_size, memory_type,
                              split_eps):
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")
    run_mr_abort_partial(cmdline_args, rma_fabric, rma_op, high_pps,
                         sl_low_latency, message_size, memory_type, split_eps)


# --- Test: incast (many initiator EPs, one target EP) ---
MR_ABORT_INCAST_NUM_DOMAINS = 4
MR_ABORT_INCAST_INITIATOR_EPS = (MR_ABORT_INCAST_NUM_DOMAINS *
                                 MR_ABORT_EPS_PER_DOMAIN)


def incast_case_params():
    """
    abort_case_params() without the 10 MiB size. The buffer footprint is
    W * ops_per_mr * message_size per side, all on one device when -D is
    used, and the incast -W (8192) needs 80 GiB at 10 MiB -- more than an
    accelerator device pool (a trn2 HBM bank is ~24 GB). The 4-endpoint
    abort test already covers 10 MiB.
    """
    return [p for p in abort_case_params() if p.values[0] != MSG_SIZE_10MIB]


def run_mr_abort_incast(cmdline_args, rma_fabric, rma_op, cancel_order,
                        ops_per_mr, high_pps, sl_low_latency, message_size,
                        memory_type, num_domains):
    """
    Incast: 16 initiator endpoints, 4 per EFA NIC across 4 NICs, all
    targeting a single endpoint on a separate platform. Skipped when
    either host has fewer than 4 EFA NICs.

    The load is per-domain: each NIC carries the same 4-endpoint,
    512-ops-per-endpoint load as the single-domain tests, so the lone
    target endpoint absorbs a 4-NIC incast.

    Only the initiator close is exercised: a high incast only makes sense
    to cancel on the transmit side.
    """
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("mr_abort incast test requires two platforms")

    if num_domains < MR_ABORT_INCAST_NUM_DOMAINS:
        pytest.skip(f"mr_abort incast test requires at least "
                    f"{MR_ABORT_INCAST_NUM_DOMAINS} EFA NICs")

    command = (f"fi_mr_abort -T abort -o {rma_op} -C {cancel_order}"
               f" -R initiator -N {ops_per_mr}"
               f" -W {mr_abort_num_mrs(ops_per_mr, MR_ABORT_INCAST_INITIATOR_EPS)}"
               f" -S {message_size}"
               f" --num-initiator-eps {MR_ABORT_INCAST_INITIATOR_EPS}"
               f" --num-target-eps 1"
               f" --eps-per-domain {MR_ABORT_EPS_PER_DOMAIN}"
               f" -I 3")
    if high_pps:
        assert(rma_op != "read")
        command += " --high-pps"

    if sl_low_latency:
        assert(rma_op != "read")
        assert(message_size <= SL_LOW_LATENCY_MAX_WRITE_SIZE)
        command += " --sl-low-latency"

    test = ClientServerTest(cmdline_args, command, timeout=360,
                            fabric=rma_fabric, memory_type=memory_type)
    test.run()


@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.memory_type(memory_type_list_symm, prefer_accelerator=True)
@pytest.mark.parametrize(
    "message_size, rma_op, high_pps, sl_low_latency, cancel_order, ops_per_mr",
    incast_case_params())
def test_mr_abort_incast_efa_direct(cmdline_args, rma_fabric, rma_op,
                                    cancel_order, ops_per_mr, high_pps,
                                    sl_low_latency, message_size, memory_type,
                                    num_domains):
    run_mr_abort_incast(cmdline_args, rma_fabric, rma_op, cancel_order,
                        ops_per_mr, high_pps, sl_low_latency, message_size,
                        memory_type, num_domains)


@pytest.mark.functional
@pytest.mark.fabric(params=["efa"])
@pytest.mark.memory_type(memory_type_list_symm)
@pytest.mark.parametrize(
    "message_size, rma_op, high_pps, sl_low_latency, cancel_order, ops_per_mr",
    incast_case_params())
def test_mr_abort_incast_efa(cmdline_args, rma_fabric, rma_op, cancel_order,
                             ops_per_mr, high_pps, sl_low_latency,
                             message_size, memory_type, num_domains):
    # No SHM guard needed: the incast helper already requires two platforms.
    run_mr_abort_incast(cmdline_args, rma_fabric, rma_op, cancel_order,
                        ops_per_mr, high_pps, sl_low_latency, message_size,
                        memory_type, num_domains)


def determine_settings_for_proto(protocol, memory_type, fabric):
    """
    Return the (env, message_size) needed to exercise a specific EFA RDM
    two-sided wire protocol with fi_mr_abort.

    Deterministic protocol pinning via env vars only applies to the
    host-memory RDM path (fabric == "efa" and host_to_host). For
    efa-direct there is no RDM protocol selection at all. In those cases
    we fall back to the previous behavior which was to pick a small, medium
    and large message size.

    To pin a single protocol on host+efa we disable the competing paths via
    env vars rather than relying on default thresholds:

      - FI_EFA_USE_DEVICE_RDMA=0 removes the read-base branch entirely, so a
        size-based choice among EAGER/MEDIUM/LONGCTS is unambiguous.
      - FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE bounds MEDIUM.
      - FI_EFA_INTER_MIN_READ_MESSAGE_SIZE bounds the read-base entry.
      - FI_EFA_RUNT_SIZE selects RUNTREAD (>0) vs LONGREAD (0), and its
        magnitude vs message_size selects runt-only (NOREAD) vs runt+read.

    :param protocol: one of EAGER, MEDIUM, LONGCTS, RUNTREAD-LONGREAD,
                     RUNTREAD-NOREAD
    :param memory_type_symm: symmetric memory type, e.g. "host_to_host",
                     "cuda_to_cuda"
    :param fabric: "efa" or "efa-direct"
    :return: (env_str, message_size) where env_str is a space-separated
             "VAR=val ..." string passed to both peers via additional_env
             (empty for the fallback path), and message_size is the -S value.
    """
    # Host-memory defaults (efa.h): eager_max ~= MTU - headers (~8 KB),
    # max_medium_msg_size = 65536, min_read_msg_size = 1048576.
    EAGER_SIZE = MSG_SIZE_4KIB            # < eager_max -> EAGER
    MEDIUM_SIZE = MSG_SIZE_64KIB          # 8 MTU-sized packets, == default medium_max -> MEDIUM
    LARGE_SIZE = MSG_SIZE_1MIB            # 1 MiB, > medium_max -> LONGCTS / read-base
    RUNT_ONLY_SIZE = MSG_SIZE_128KIB      # <= runt budget so no trailing READ is posted

    # Representative size per protocol, used both for the host+efa pinned
    # path and the fallback path.
    proto_size = {
        "EAGER": EAGER_SIZE,
        "MEDIUM": MEDIUM_SIZE,
        "LONGCTS": LARGE_SIZE,
        "LONGREAD": LARGE_SIZE,
        "RUNTREAD-LONGREAD": LARGE_SIZE,
        "RUNTREAD-NOREAD": RUNT_ONLY_SIZE,
    }
    if protocol not in proto_size:
        raise ValueError(f"unknown protocol: {protocol}")

    # Fallback: HMEM or efa-direct. Protocol pinning via the host RDM
    # thresholds does not apply; return a representative size and no env.
    if fabric != "efa" or memory_type != "host_to_host":
        return "", proto_size[protocol]

    # Host + efa: pin the protocol deterministically via env-var thresholds.
    # The selected protocol can be confirmed manually by running with
    # FI_LOG_LEVEL=debug and looking for the provider's
    # "efa-rdm selecting transfer protocol ..." log line.
    if protocol == "EAGER":
        # 4 KB is below eager_max, and below the 1 MiB host min_read
        # threshold so read-base is never considered. No threshold override.
        proto_env = ""
    elif protocol == "MEDIUM":
        # 64 KB == the default host max_medium_msg_size (65536), which spans
        # ~8 MTU-sized REQ packets. It is above eager_max and below min_read
        # (1 MiB), and the selection test is total_len <= max_medium_msg_size,
        # so the defaults select MEDIUM. No threshold override.
        proto_env = ""
    elif protocol == "LONGCTS":
        # Disable read-base; size above medium_max forces LONGCTS.
        proto_env = ("FI_EFA_USE_DEVICE_RDMA=0 "
                     "FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE=65536")
    elif protocol == "LONGREAD":
        # Pure long read: read-base enabled (USE_DEVICE_RDMA=1) with the
        # runt budget set to 0, so no runt segments ride the RTM and the
        # entire payload is pulled by the receiver via RDMA READ. The
        # LONGREAD RTM is a pure read-request (no source-MR payload), so
        # closing the source MR cannot flush it before the receiver matches:
        # the recv is always matched and its tail READ then fails against
        # the invalidated source rkey, yielding exactly one FI_ECANCELED.
        # This is the only protocol owed a completion under -X (see
        # abort_owes_rx_completion). Run with -H (HOMOGENEOUS_PEERS) so the
        # sender does not stall on a handshake before selecting it.
        proto_env = ("FI_EFA_USE_DEVICE_RDMA=1 "
                     "FI_EFA_RUNT_SIZE=0 "
                     "FI_EFA_INTER_MIN_READ_MESSAGE_SIZE=65536")
    elif protocol == "RUNTREAD-LONGREAD":
        # Read-base enabled with a runt budget smaller than the message
        # (RUNT_SIZE=131072 < 1 MiB). Under the fi_mr_abort flood this
        # exercises BOTH read-base sub-protocols in one run: head-of-line
        # messages select RUNTREAD (runt segments + a trailing RDMA READ)
        # while the per-peer runt budget and num_read_msg_in_flight==0
        # gates pass; once the budget drains / reads are in flight,
        # selection degrades to LONGREAD. Combining them in a single stage
        # mirrors how a real read-base workload mixes the two.
        proto_env = ("FI_EFA_USE_DEVICE_RDMA=1 "
                     "FI_EFA_RUNT_SIZE=131072 "
                     "FI_EFA_INTER_MIN_READ_MESSAGE_SIZE=65536")
    elif protocol == "RUNTREAD-NOREAD":
        # RUNTREAD-NOREAD: runt-only runting read: the whole message fits in
        # the runt budget (total_len <= runt_size), so all data rides REQ
        # packets and no RDMA READ is posted. runt_size >= msg >= min_read.
        proto_env = ("FI_EFA_USE_DEVICE_RDMA=1 "
                     "FI_EFA_RUNT_SIZE=262144 "
                     "FI_EFA_INTER_MIN_READ_MESSAGE_SIZE=65536")
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    return proto_env.strip(), proto_size[protocol]


def abort_owes_rx_completion(protocol):
    """
    Whether an aborted send/tagged message is owed a terminal recv
    completion on the target for this protocol (the -X flag of
    fi_mr_abort).

    Owed: LONGREAD only. The LONGREAD RTM is a pure read-request control
    packet -- it carries the read iov, no inline user data, and its send WR
    uses the internal TX pool MR rather than the user's source MR. So
    closing the source MR cannot flush the RTM: it is always delivered, the
    receiver always matches the recv and posts the tail RDMA READ, and that
    READ then fails against the invalidated source rkey -- driving the
    matched rxe to a clean FI_ECANCELED. Exactly one terminal completion is
    therefore guaranteed.

    Not owed: EAGER, MEDIUM, LONGCTS, RUNTREAD-LONGREAD, RUNTREAD-NOREAD.
    Every one of these carries source-MR user data in its RTM (EAGER/MEDIUM
    full or first segment, LONGCTS first segment, RUNTREAD runt segments),
    so the RTM itself can be flushed or gen-check cancelled before the
    receiver ever matches the recv. When that happens the receiver owes no
    completion, so the recv-completion count is indeterminate and -X (which
    blocks until reaped == required) would hang the target. These use the
    slack path: a stray FI_ECANCELED is tolerated but never required.
    """
    return protocol in ("LONGREAD",)


# --- Test: send and tagged ---
SEND_PROTOCOLS = ["EAGER", "MEDIUM", "LONGCTS", "LONGREAD",
                  "RUNTREAD-LONGREAD", "RUNTREAD-NOREAD"]

# efa-direct max send size is 8KB
EFA_DIRECT_MAX_SEND_SIZE = MSG_SIZE_8KIB


def send_case_params(fabric):
    """
    Generate the (tagged, protocol) combinations that can run on the given
    fabric, so no never-runnable case is collected: efa-direct does not
    support tagged sends and its 8KB send limit leaves only EAGER; the efa
    fabric runs the full cross product.
    """
    params = []
    for tagged in [False, True]:
        for protocol in SEND_PROTOCOLS:
            if fabric == "efa-direct":
                if tagged:
                    continue
                _, message_size = determine_settings_for_proto(
                    protocol, "host_to_host", "efa-direct")
                if message_size > EFA_DIRECT_MAX_SEND_SIZE:
                    continue
            params.append(pytest.param(tagged, protocol,
                                       id=f"tagged_{tagged}-{protocol}"))
    return params


def run_mr_abort_send(cmdline_args, fabric, cancel_order, close_side,
                      ops_per_mr, tagged, protocol, memory_type):
    send_op = "tagged" if tagged else "send"

    env, message_size = determine_settings_for_proto(protocol, memory_type, fabric)

    owe_flag = " -X" if abort_owes_rx_completion(protocol) else ""
    # LONGREAD is an extra-feature (read-based) protocol. Pass -H to set
    # HOMOGENEOUS_PEERS on the endpoint, which makes it ignore the handshake
    # requirement before selecting a read-based protocol. Without it the
    # first sends are queued pending the peer handshake and the abort race is
    # nondeterministic; skipping the handshake pins LONGREAD from the first
    # send so the target reliably owes -- and can enforce via -X -- exactly
    # one completion per send.
    homogeneous_flag = " -H" if protocol == "LONGREAD" else ""
    command = (f"fi_mr_abort -T {send_op} -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W {mr_abort_num_mrs(ops_per_mr)}"
               f" -S {message_size}{owe_flag}{homogeneous_flag}"
               f"{MR_ABORT_EP_ARGS}  -A ep_first")
    test = ClientServerTest(cmdline_args, command, timeout=360, fabric=fabric,
                            memory_type=memory_type, additional_env=env)
    test.run()


@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"])
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
# TODO add "target" once efa supports canceling posted RX buffers
@pytest.mark.parametrize("close_side", ["initiator"])
@pytest.mark.parametrize("ops_per_mr", [1, 4])
@pytest.mark.parametrize("tagged, protocol", send_case_params("efa-direct"))
@pytest.mark.memory_type(memory_type_list_symm, prefer_accelerator=True)
def test_mr_abort_send_efa_direct(cmdline_args, fabric, cancel_order,
                                  close_side, ops_per_mr, tagged, protocol,
                                  memory_type):
    run_mr_abort_send(cmdline_args, fabric, cancel_order, close_side,
                      ops_per_mr, tagged, protocol, memory_type)


@pytest.mark.functional
@pytest.mark.fabric(params=["efa"])
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
# TODO add "target" once efa supports canceling posted RX buffers
@pytest.mark.parametrize("close_side", ["initiator"])
@pytest.mark.parametrize("ops_per_mr", [1, 4])
@pytest.mark.parametrize("tagged, protocol", send_case_params("efa"))
@pytest.mark.memory_type(memory_type_list_symm)
def test_mr_abort_send_efa(cmdline_args, fabric, cancel_order, close_side,
                           ops_per_mr, tagged, protocol, memory_type):
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")
    run_mr_abort_send(cmdline_args, fabric, cancel_order, close_side,
                      ops_per_mr, tagged, protocol, memory_type)

