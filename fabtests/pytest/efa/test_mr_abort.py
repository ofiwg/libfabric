import pytest
from common import ClientServerTest

# fi_mr_abort fabtest will allocate MR_ABORT_NUM_MRS and attempt
# to post N transfers per MR until the provider returns -FI_EAGAIN
# or we posted transactions for each MR. A larger number corresponds
# to increased load on the NIC and increased memory preassure on the
# instance.
MR_ABORT_NUM_MRS = 2046


# --- Test: abort (RMA) ---
@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"])  # TODO add test for efa fabric
@pytest.mark.parametrize("rma_op", ["write", "read", "writedata"])
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
@pytest.mark.parametrize("close_side", ["initiator", "target"])
@pytest.mark.parametrize("ops_per_mr", [1, 4])
@pytest.mark.parametrize("message_size", [
    4096,
    65536,
    1048576,
    # 10 MiB: large transfers consume far more memory/NIC resources, so
    # run this size serially to avoid resource contention with parallel
    # workers.
    pytest.param(10485760, marks=pytest.mark.serial),
])
def test_mr_abort(cmdline_args, fabric, rma_op, cancel_order, close_side, ops_per_mr, message_size, memory_type_symm):
    if fabric == "efa" and cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")

    if message_size == 10485760 and (cancel_order == 'random' or ops_per_mr == 4):
        pytest.skip("fi_mr_abort 10MB test only runs with reverse cancel order on 1 operation per MR to save run time")

    command = (f"fi_mr_abort -T abort -o {rma_op} -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W {MR_ABORT_NUM_MRS}"
               f" -S {message_size}")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric, memory_type=memory_type_symm)
    test.run()


# --- Test: partial (2 MRs on same buffer) ---
@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"]) # TODO add test for efa fabric
@pytest.mark.parametrize("rma_op", ["write", "read", "writedata"])
@pytest.mark.parametrize("message_size", [
    4096,
    65536,
    1048576,
    # 10 MiB: run serially to avoid resource contention (see test_mr_abort).
    pytest.param(10485760, marks=pytest.mark.serial),
])
def test_mr_abort_partial(cmdline_args, fabric, rma_op, message_size, memory_type_symm):
    if fabric == "efa" and cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")

    command = (f"fi_mr_abort -T partial -o {rma_op} -S {message_size}")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric, memory_type=memory_type_symm)
    test.run()


def determine_settings_for_proto(protocol, memory_type_symm, fabric):
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
        "RUNTREAD-LONGREAD": LARGE_SIZE,
        "RUNTREAD-NOREAD": RUNT_ONLY_SIZE,
    }
    if protocol not in proto_size:
        raise ValueError(f"unknown protocol: {protocol}")

    # Fallback: HMEM or efa-direct. Protocol pinning via the host RDM
    # thresholds does not apply; return a representative size and no env.
    if fabric != "efa" or memory_type_symm != "host_to_host":
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

    Owed: LONGCTS and RUNTREAD-LONGREAD. The receiver matches the recv
    and takes partial data (a CTS handshake, or a runt + tail RDMA READ)
    before the abort, so the provider must complete that matched rxe with
    a clean FI_ECANCELED.

    Not owed: EAGER, MEDIUM, RUNTREAD-NOREAD (runt-only). An aborted
    message either delivered nothing or rides REQ packets with no CTS /
    no READ; the receiver is not required to produce a completion (a stray
    FI_ECANCELED may still arrive and is tolerated, but is not required).
    """
    return protocol in ("LONGCTS", "RUNTREAD-LONGREAD")


# --- Test: send ---
@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"]) # TODO add test for efa fabric
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
@pytest.mark.parametrize("close_side", ["initiator"]) # TODO add target
@pytest.mark.parametrize("ops_per_mr", [1, 4])
@pytest.mark.parametrize("protocol", ["EAGER", "MEDIUM", "LONGCTS", "RUNTREAD-LONGREAD", "RUNTREAD-NOREAD"])
def test_mr_abort_send(cmdline_args, fabric, cancel_order, close_side,
                       ops_per_mr, protocol, memory_type_symm):
    if fabric == "efa" and cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")

    if close_side == "target":
        pytest.skip("efa does not currently support canceling posted RX buffers")

    env, message_size = determine_settings_for_proto(protocol, memory_type_symm, fabric)

    # read-base (LONGREAD/RUNTREAD) and the larger LONGCTS sizes require RDMA
    # read / message sizes that efa-direct does not support for sends.
    if fabric == "efa-direct" and message_size > 8192:
        pytest.skip("efa-direct max send size is 8KB")

    owe_flag = " -X" if abort_owes_rx_completion(protocol) else ""
    command = (f"fi_mr_abort -T send -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W {MR_ABORT_NUM_MRS}"
               f" -S {message_size}{owe_flag}  -A ep_first")
    test = ClientServerTest(cmdline_args, command, timeout=360, fabric=fabric,
                            memory_type=memory_type_symm, additional_env=env)
    test.run()


# --- Test: tagged ---
@pytest.mark.functional
@pytest.mark.fabric(params=["efa-direct"])  # TODO add test for efa fabric
@pytest.mark.parametrize("cancel_order", ["reverse", "random"])
@pytest.mark.parametrize("close_side", ["initiator"]) # TODO add target
@pytest.mark.parametrize("ops_per_mr", [1, 4])
@pytest.mark.parametrize("protocol", ["EAGER", "MEDIUM", "LONGCTS", "RUNTREAD-LONGREAD", "RUNTREAD-NOREAD"])
def test_mr_abort_tagged(cmdline_args, fabric, cancel_order, close_side,
                         ops_per_mr, protocol, memory_type_symm):
    if fabric == "efa" and cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("fi_mr_abort not supported with efa with SHM")

    if fabric == "efa-direct":
        pytest.skip("efa-direct does not support tagged messages")

    if close_side == "target":
        pytest.skip("efa does not currently support canceling posted RX buffers")

    env, message_size = determine_settings_for_proto(protocol, memory_type_symm, fabric)

    owe_flag = " -X" if abort_owes_rx_completion(protocol) else ""
    command = (f"fi_mr_abort -T tagged -C {cancel_order}"
               f" -R {close_side} -N {ops_per_mr} -W {MR_ABORT_NUM_MRS}"
               f" -S {message_size}{owe_flag} -A ep_first")
    test = ClientServerTest(cmdline_args, command, timeout=300, fabric=fabric,
                            memory_type=memory_type_symm, additional_env=env)
    test.run()
