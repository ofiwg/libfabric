# Debugging EFA RDM Hangs with State Dump

## Overview

The EFA RDM provider includes a signal-triggered state dumper that prints
the internal state of all endpoints and peers when a process receives a
configured signal. This is designed for diagnosing hangs where the
application is busy-polling with no forward progress.

## Enabling

Set `FI_EFA_STATE_DUMP_SIGNAL` to the signal number you want to trigger
the dump. Default is 0 (disabled).

```bash
# Use SIGUSR2 (signal 12)
export FI_EFA_STATE_DUMP_SIGNAL=12

# Use SIGUSR1 (signal 10)
export FI_EFA_STATE_DUMP_SIGNAL=10
```

## Usage

```bash
# Find the hung process
ps aux | grep <application_name>

# Trigger the state dump (assuming FI_EFA_STATE_DUMP_SIGNAL=12)
kill -12 <pid>
```

The dump is printed to stderr at `FI_LOG_LEVEL=warn` (or higher).

## Output Format

```
=== EFA RDM EP STATE DUMP ===
--- EP-level counters ---
efa_outstanding_tx_ops=3 efa_rnr_queued_pkt_cnt=0 efa_rx_pkts_posted=512 efa_rx_pkts_held=2 efa_max_outstanding_tx_ops=8192 efa_max_outstanding_rx_ops=8192
queued_copy_num=0 ope_queued_before_handshake_cnt=0 txe_cnt=2 rxe_cnt=1
--- Peer-level state ---
  PEER 0x5555deadbeef fi_addr=0 flags=0x4 exp_msg_id=10 next_msg_id=12 txe_cnt=0 rxe_cnt=0 overflow_cnt=0 rnr_queued=0 backoff_wait=0 us outstanding_tx=0
  PEER 0x5555cafebabe fi_addr=1 flags=0x4 exp_msg_id=42 next_msg_id=45 txe_cnt=1 rxe_cnt=0 overflow_cnt=0 rnr_queued=0 backoff_wait=0 us outstanding_tx=1
    TXE peer=0x5555cafebabe fi_addr=1 msg_id=44 op=2 total_len=1310720 bytes_sent=0 window=0 internal_flags=0x0
=== END EFA RDM EP STATE DUMP ===
```

All peers are dumped regardless of whether they have outstanding operations.
The `peer=` pointer in TXE/RXE lines can be used to cross-reference with the
`PEER` line above.

In debug builds, queued packet entries are additionally printed under each
TXE/RXE via `efa_rdm_pke_print()`.

## Interpreting the Dump

### Endpoint-Level Fields

| Field | Meaning |
|-------|---------|
| `efa_outstanding_tx_ops` | Number of send/read operations submitted to EFA device but not yet completed |
| `efa_rnr_queued_pkt_cnt` | Packets queued due to RNR (Receiver Not Ready) errors |
| `efa_rx_pkts_posted` | Receive buffers posted to the EFA device |
| `efa_rx_pkts_held` | RX packets held by progress engine (pending copy or local read) |
| `queued_copy_num` | Number of queued memory copies (HMEM) |
| `ope_queued_before_handshake_cnt` | Operations waiting for peer handshake |
| `txe_cnt` | Total outstanding TX operations across all peers |
| `rxe_cnt` | Total outstanding RX operations across all peers |

### Per-Peer Fields

| Field | Meaning |
|-------|---------|
| `PEER <ptr>` | Pointer to the peer structure (for cross-referencing with TXE/RXE) |
| `fi_addr` | Fabric address of the peer |
| `flags` | Peer state flags (see below) |
| `exp_msg_id` | Next expected message ID in the reorder buffer |
| `next_msg_id` | Next message ID to assign when sending to this peer |
| `txe_cnt` | Outstanding send operations to this peer |
| `rxe_cnt` | Outstanding receive operations from this peer |
| `overflow_cnt` | Messages in the overflow list (exceeded reorder window) |
| `rnr_queued` | Packets queued for RNR retry to this peer |
| `backoff_wait` | Current RNR backoff wait time in microseconds |
| `outstanding_tx` | TX operations outstanding on EFA device for this peer |

### Peer Flags

| Flag | Hex | Meaning |
|------|-----|---------|
| `EFA_RDM_PEER_REQ_SENT` | 0x1 | Initial request sent to peer |
| `EFA_RDM_PEER_HANDSHAKE_SENT` | 0x2 | Handshake packet sent |
| `EFA_RDM_PEER_HANDSHAKE_RECEIVED` | 0x4 | Handshake received from peer |
| `EFA_RDM_PEER_IN_BACKOFF` | 0x8 | Peer is in RNR backoff mode |

### Per-Peer Operation Fields (TXE/RXE)

| Field | Meaning |
|-------|---------|
| `peer` | Pointer to the peer structure (matches `PEER <ptr>` above) |
| `fi_addr` | Fabric address of the peer |
| `msg_id` | Message sequence number |
| `op` | Operation type (2=msg, 3=tagged, etc.) |
| `total_len` | Total message length in bytes |
| `bytes_sent` | Bytes transmitted so far |
| `window` | CTS credit window remaining (long-CTS protocol) |
| `internal_flags` | Queued state flags (see below) |

### Operation Internal Flags

| Flag | Meaning |
|------|---------|
| `[QUEUED_RNR]` | Operation queued due to RNR, waiting for backoff to expire |
| `[QUEUED_CTRL]` | Control packet queued (e.g., CTS, EOR) |
| `[QUEUED_READ]` | RDMA read operation queued (waiting for resources) |
| `[QUEUED_HANDSHAKE]` | Operation waiting for handshake to complete |

## Common Hang Patterns

### Pattern 1: Waiting for CTS (Long-CTS Protocol)

```
  PEER 0x5555cafebabe fi_addr=1 flags=0x4 exp_msg_id=100 next_msg_id=101 txe_cnt=1 ...
    TXE peer=0x5555cafebabe fi_addr=1 msg_id=100 op=2 total_len=1310720 bytes_sent=0 window=0 internal_flags=0x0
```

**Diagnosis**: Sender has a TXE with `bytes_sent=0` and `window=0`. This
means the sender sent a LONGCTS_RTM request and is waiting for the
receiver to reply with a CTS (Clear To Send) packet. The CTS never
arrived.

**Root cause**: CTS packet lost at the SRD fabric level, or receiver
never processed the RTM (check receiver's dump).

### Pattern 2: Waiting for RDMA Read Completion (Long-Read Protocol)

```
  PEER 0x5555cafebabe fi_addr=1 flags=0x4 exp_msg_id=100 next_msg_id=101 txe_cnt=1 ...
    TXE peer=0x5555cafebabe fi_addr=1 msg_id=100 op=2 total_len=1310720 bytes_sent=1310720 window=0 internal_flags=0x0
```

On the receiver side:
```
  PEER 0x5555deadbeef fi_addr=0 flags=0x4 ... rxe_cnt=1 ...
    RXE peer=0x5555deadbeef fi_addr=0 msg_id=100 op=2 total_len=1310720 bytes_sent=0 window=1310720 internal_flags=0x0
```

**Diagnosis**: Sender shows `bytes_sent=total_len` (RTM sent with memory
key). Receiver has an RXE with `window > 0` indicating it issued an RDMA
read but the read never completed.

**Root cause**: RDMA read operation failed silently at the fabric level,
or the EOR (End Of Read) packet from receiver to sender was lost.

### Pattern 3: RNR Backoff Storm

```
  PEER 0x5555cafebabe fi_addr=1 flags=0xc ... rnr_queued=15 backoff_wait=1000000 us outstanding_tx=0
    TXE peer=0x5555cafebabe fi_addr=1 msg_id=50 op=2 total_len=65536 bytes_sent=0 window=0 internal_flags=0x200 [QUEUED_RNR]
```

**Diagnosis**: Peer is in backoff (`flags` has `0x8`), with packets
queued for RNR retry. The `backoff_wait` shows how long the backoff
period is (1 second in this example).

**Root cause**: Receiver's RQ was full when the sender tried to send.
With exponential backoff, wait times can grow large. If the receiver
never posts new receive buffers, this becomes a permanent hang.

**Action**: Check if `efa_rx_pkts_posted` on the receiver is 0 or very
low. Increase `FI_EFA_RX_SIZE` if needed.

### Pattern 4: Handshake Deadlock

```
  PEER 0x5555cafebabe fi_addr=1 flags=0x2 ... txe_cnt=1 ...
    TXE peer=0x5555cafebabe fi_addr=1 msg_id=0 op=2 total_len=1310720 bytes_sent=0 window=0 internal_flags=0x4000 [QUEUED_HANDSHAKE]
```

**Diagnosis**: Operation is queued waiting for handshake (`flags=0x2`
means handshake sent but not received back). The peer hasn't responded
to the handshake.

**Root cause**: Handshake packet lost, or peer hasn't started yet, or
peer's RQ is full (RNR on handshake).

### Pattern 5: Receive Window Overflow

```
  PEER 0x5555cafebabe fi_addr=1 flags=0x4 exp_msg_id=42 next_msg_id=100 ... overflow_cnt=50 ...
```

**Diagnosis**: Large gap between `exp_msg_id` (42) and `next_msg_id`
(100) with many entries in the overflow list. The receiver is waiting
for message 42 which never arrived, blocking all subsequent messages.

**Root cause**: Message 42 was lost. All messages 43-99+ are buffered
but can't be processed until 42 arrives.

**Action**: Increase `FI_EFA_RECVWIN_SIZE` to buffer more out-of-order
messages while investigating the packet loss.

### Pattern 6: No Outstanding Operations (Idle Hang)

```
=== EFA RDM EP STATE DUMP ===
--- EP-level counters ---
efa_outstanding_tx_ops=0 efa_rnr_queued_pkt_cnt=0 ... txe_cnt=0 rxe_cnt=0
--- Peer-level state ---
  PEER 0x5555deadbeef fi_addr=0 flags=0x4 exp_msg_id=10 next_msg_id=10 txe_cnt=0 rxe_cnt=0 overflow_cnt=0 rnr_queued=0 backoff_wait=0 us outstanding_tx=0
=== END EFA RDM EP STATE DUMP ===
```

**Diagnosis**: No outstanding operations on any peer. The EFA provider
has nothing pending.

**Root cause**: The hang is above the provider level — likely in the
MPI layer or application. The provider completed all its work but the
upper layer is waiting for something else (e.g., a message from a peer
that the peer never sent because *it* is hung).

**Action**: Check the dump on the peer side. One side will show
outstanding operations.

## Multi-Rank Debugging

For MPI hangs, dump all ranks:

```bash
# Dump all MPI processes on this node (assuming FI_EFA_STATE_DUMP_SIGNAL=12)
for pid in $(pgrep -f "my_application"); do
    echo "=== PID $pid ===" >&2
    kill -12 $pid
    sleep 0.1
done
```

Then correlate: find the rank with outstanding TXE/RXE and use the
`peer=` pointer and `fi_addr` to identify which pair of ranks is stuck.

## Environment Variables

| Variable | Effect on Dump |
|----------|---------------|
| `FI_EFA_STATE_DUMP_SIGNAL=<N>` | Signal number to trigger dump (0=disabled, 10=SIGUSR1, 12=SIGUSR2) |
| `FI_LOG_LEVEL=warn` | Minimum level to see dump output (default) |
| `FI_EFA_RECVWIN_SIZE=N` | Changes reorder buffer size (affects overflow_cnt) |
| `FI_EFA_MAX_TIMEOUT=N` | Caps RNR backoff wait time in microseconds (affects backoff_wait) |
