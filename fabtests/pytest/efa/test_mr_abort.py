import pytest
from common import ClientServerTest
from efa.efa_common import memory_type_list_symm


pytestmark = pytest.mark.pre_release


# fi_mr_abort fabtest will allocate MR_ABORT_NUM_MRS and attempt
# to post N transfers per MR until the provider returns -FI_EAGAIN
# or we posted transactions for each MR. A larger number corresponds
# to increased load on the NIC and increased memory preassure on the
# instance.
MR_ABORT_NUM_MRS = 2046

BASE_MESSAGE_SIZES = [
    64,
    # 128B is the largest write WQE the low latency service level
    # accelerates; only low-latency cases are generated at this size
    128,
    4096,
    65536,
    1048576,
    10485760,
]

# The high PPS WQE hint only accelerates writes <= 8KB, but writes up to
# 64KB marked high-pps exercise special handling that routes them back to
# the default processing path. Request high PPS for writes up to 64KB to
# utilize both paths; larger sizes add no new coverage.
HIGH_PPS_MAX_WRITE_SIZE = 65536

# The low latency service level only accelerates write WQEs <= 128 bytes,
# but request it for writes up to 8 KiB so a single run mixes the
# accelerated (<= 128B) and non-accelerated flows.
SL_LOW_LATENCY_MAX_WRITE_SIZE = 8192


def rma_case_params():
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
            if size == 10485760:
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
                    if size == 128 and not use_sl_low_latency:
                        continue

                    id = f"{rma_op}-{size}"

                    if use_high_pps is not None:
                        id += f"-high_pps-{use_high_pps}"

                    if use_sl_low_latency is not None:
                        id += f"-sl_low_lat-{use_sl_low_latency}"

                    yield pytest.param(size, rma_op, use_high_pps,
                                       use_sl_low_latency, marks=marks, id=id)


def abort_case_params():
    """
    Expand rma_case_params() with the (cancel_order, ops_per_mr)
    combinations that actually run for each message size.

    The 10 MiB size only runs with reverse cancel order on 1 operation per
    MR to save run time; every other size runs the full cross product.
    """
    for p in rma_case_params():
        size = p.values[0]
        if size == 10485760:
            cases = [("reverse", 1)]
        else:
            cases = [(cancel_order, ops_per_mr)
                     for cancel_order in ("reverse", "random")
                     for ops_per_mr in (1, 4)]
        for cancel_order, ops_per_mr in cases:
            yield pytest.param(*p.values, cancel_order, ops_per_mr,
                               marks=p.marks,
                               id=f"{p.id}-{cancel_order}-ops_{ops_per_mr}")


# --- Test: abort (RMA) ---
@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"])  # TODO add test for efa fabric
@pytest.mark.parametrize("close_side", ["initiator", "target"])
@pytest.mark.memory_type(memory_type_list_symm, prefer_accelerator=True)  # TODO run both system + hmem memory cases on the efa fabric
@pytest.mark.parametrize(
    "message_size, rma_op, high_pps, sl_low_latency, cancel_order, ops_per_mr",
    list(abort_case_params()))
def test_mr_abort(cmdline_args, rma_fabric, rma_op, cancel_order, close_side, ops_per_mr,
                  high_pps, sl_low_latency, message_size, memory_type):
    if rma_fabric == "efa" and cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")

    command = (f"fi_mr_abort -T abort -o {rma_op} -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W {MR_ABORT_NUM_MRS}"
               f" -S {message_size}")
    if high_pps:
        assert(rma_op != "read")
        command += " --high-pps"

    if sl_low_latency:
        assert(rma_op != "read")
        assert(message_size <= SL_LOW_LATENCY_MAX_WRITE_SIZE)
        command += " --sl-low-latency"

    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=rma_fabric, memory_type=memory_type)
    test.run()


# --- Test: partial (2 MRs on same buffer) ---
@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"]) # TODO add test for efa fabric
@pytest.mark.memory_type(memory_type_list_symm, prefer_accelerator=True)  # TODO run both system + hmem memory cases on the efa fabric
@pytest.mark.parametrize("message_size, rma_op, high_pps, sl_low_latency",
                         list(rma_case_params()))
def test_mr_abort_partial(cmdline_args, rma_fabric, rma_op, high_pps,
                          sl_low_latency, message_size, memory_type):
    if rma_fabric == "efa" and cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")

    command = (f"fi_mr_abort -T partial -o {rma_op} -S {message_size}")

    if high_pps:
        assert(rma_op != "read")
        command += " --high-pps"

    if sl_low_latency:
        assert(rma_op != "read")
        assert(message_size <= SL_LOW_LATENCY_MAX_WRITE_SIZE)
        command += " --sl-low-latency"

    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=rma_fabric, memory_type=memory_type)
    test.run()


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
    EAGER_SIZE = 4096            # < eager_max -> EAGER
    MEDIUM_SIZE = 65536          # 8 MTU-sized packets, == default medium_max -> MEDIUM
    LARGE_SIZE = 1048576         # 1 MiB, > medium_max -> LONGCTS / read-base
    RUNT_ONLY_SIZE = 131072      # <= runt budget so no trailing READ is posted

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
EFA_DIRECT_MAX_SEND_SIZE = 8192


def send_case_params():
    """
    Generate the (tagged, protocol) combinations that can actually run on
    the fabrics generated by the fabric marker (efa-direct only): efa-direct
    does not support tagged sends, and its 8KB send limit leaves only
    EAGER (4KB). The tagged dimension is kept for the efa fabric.

    TODO take the fabric into account once the efa fabric is added; tagged
    sends and the larger protocols apply there.
    """
    for tagged in [False]:
        for protocol in SEND_PROTOCOLS:
            # memory_type is irrelevant for the size lookup on efa-direct:
            # any non-("efa", host_to_host) combination takes the fallback
            # path in determine_settings_for_proto.
            _, message_size = determine_settings_for_proto(
                protocol, "host_to_host", "efa-direct")
            if message_size > EFA_DIRECT_MAX_SEND_SIZE:
                continue
            yield pytest.param(tagged, protocol,
                               id=f"tagged_{tagged}-{protocol}")


@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"]) # TODO add test for efa fabric
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
# TODO add "target" once efa supports canceling posted RX buffers
@pytest.mark.parametrize("close_side", ["initiator"])
@pytest.mark.parametrize("ops_per_mr", [1, 4])
@pytest.mark.parametrize("tagged, protocol", list(send_case_params()))
@pytest.mark.memory_type(memory_type_list_symm, prefer_accelerator=True)  # TODO run both system + hmem memory cases on the efa fabric
def test_mr_abort_send(cmdline_args, fabric, cancel_order, close_side,
                       ops_per_mr, tagged, protocol, memory_type):
    if fabric == "efa" and cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")

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
               f" -R {close_side} -N {ops_per_mr} -W {MR_ABORT_NUM_MRS}"
               f" -S {message_size}{owe_flag}{homogeneous_flag}  -A ep_first")
    test = ClientServerTest(cmdline_args, command, timeout=360, fabric=fabric,
                            memory_type=memory_type, additional_env=env)
    test.run()

