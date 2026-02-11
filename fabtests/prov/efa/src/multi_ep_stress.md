# Multi-Endpoint Stress Test

## Overview

The `multi_ep_stress.c` test validates libfabric's EFA provider under stress conditions involving multiple endpoints, endpoint recycling, and concurrent worker threads. It simulates real-world scenarios where applications dynamically create/destroy endpoints while maintaining active communication.

## Test Architecture

### Roles

The test operates in a client-server model with two distinct roles:

- **Sender**: Initiates data transfer operations to receivers
- **Receiver**: Accepts incoming data from senders

Role is determined by the presence of `dst_addr` command-line argument (sender if provided, receiver otherwise).

### Worker Model

Both senders and receivers use a multi-threaded worker pool architecture:

- Each worker manages its own endpoint lifecycle
- Workers are assigned specific peer connections based on distribution logic
- Workers operate independently with thread-safe coordination

## Key Features

### 1. Endpoint Recycling

The core stress mechanism involves repeatedly destroying and recreating endpoints during active communication:

- **Sender EP Cycles**: Controlled by `--sender-ep-cycles` (default: 1)
- **Receiver EP Cycles**: Controlled by `--receiver-ep-cycles` (default: 1)

Within each cycle:
1. Endpoint is created with fresh CQ/AV resources (optionally it can reuse shared ones)
2. Operations are posted and completed
3. Endpoint is destroyed (optionally before all completions drain)
4. New endpoint is created for next cycle

### 2. Dynamic Address Updates

When receivers recycle endpoints, they notify senders via an out-of-band (OOB) control channel:

- Receivers send `EP_MESSAGE_TYPE_UPDATE` messages containing new endpoint addresses and RMA info
- Senders dynamically update their AV (Address Vector) by removing old addresses and inserting new ones
- Communication continues seamlessly across endpoint transitions

### 3. Resource Sharing Options

The test supports two resource sharing modes:

- **Shared AV** (`--shared-av`): Single address vector shared across all worker threads
- **Shared CQ** (`--shared-cq`): Single completion queue shared across all workers (with mutex protection)
- **Default**: Each endpoint gets dedicated AV and CQ resources

### 4. Operation Types

Three libfabric operation types are supported via `--op-type`:

- **untagged** (default): Standard `fi_send`/`fi_recv` messaging
- **tagged**: Tagged messaging using `fi_tsend`/`fi_trecv` with tag `0x123`
- **writedata**: One-sided RMA writes with immediate data using `fi_writedata`

### 5. Reproducable pseudo-random flow

Tests print unique (time-based) random seed at start.
The seed also can be specified with `--random-seed` cmdline argument.
Thread-safe random functions allow to re-run tests with the same random choises.


## Test Flow

### Initialization Phase

1. Parse command-line options and configure test parameters
2. Initialize libfabric fabric/domain resources
3. Establish OOB socket connection for control messages
4. Create shared resources (if enabled)

### Sender Workflow

For each sender worker thread:

1. **Enter EP cycle loop**:
   - Create new endpoint with CQ/AV
   - Insert all previously cached peer addresses into AV
   - Random sleep (0-100ms) to simulate real workload
2. **Message posting loop**:
   - Check for pending AV updates from receivers, apply and cache the updates
   - Post send/write operation to next peer (round-robin)
   - Handle `-FI_EAGAIN` by draining completions
   - Track operations posted/completed per cycle and per peer
3. **Cycle completion**:
   - Randomly decide whether to wait for all completions or proceed immediately
   - Destroy endpoint and start next cycle

### Receiver Workflow

For each receiver worker thread:

1. **Enter EP cycle loop**:
   - Create new endpoint with CQ/AV
   - Allocate RMA buffer (if using writedata operation)
   - Notify all connected senders of new endpoint address
   - Random sleep (0-100ms)
2. **Message posting loop**:
   - Post receive operations (for msg operations)
   - For RMA writedata, skip posting (writes are one-sided)
3. **Cycle completion**:
   - Randomly decide whether to wait for all completions or proceed immediately
   - Destroy endpoint and start next cycle
4. **Termination**: Send terminator message to control thread

### Coordination

- **Control Queue**: Thread-safe MPSC (multi-producer single-consumer) queue for endpoint updates
- **OOB Channel**: TCP socket for cross-process control messages between sender and receiver processes
- **Completion Handling**: Polling-based with timeout protection (default 10s)

## Memory Management

### Context Pool

Pre-allocated memory pool for efficient buffer and context management:

- Buffers registered as single contiguous memory region
- `fi_context2` structures allocated alongside buffers
- Supports incremental allocation without fragmentation
- Memory registration happens once per pool lifetime

### Worker Distribution

Peers are distributed across workers using modulo arithmetic:

- If `peers <= workers`: Each worker gets one peer
- If `peers > workers`: Peers distributed evenly with remainder handled

Example: 3 senders, 8 receivers → Sender 0 gets receivers [0,3,6], Sender 1 gets [1,4,7], Sender 2 gets [2,5]

## Stress Factors

The test stresses the provider through:

1. **Concurrent endpoint operations**: Multiple threads creating/destroying endpoints simultaneously
2. **Address vector churn**: Frequent AV insertions/removals during active communication
3. **Incomplete draining**: Endpoints may be destroyed before all completions are processed
4. **Resource contention**: Optional shared CQ/AV increases lock contention
5. **Random timing**: Sleep intervals create non-deterministic interleaving
6. **High message volume**: Configurable message count per endpoint lifecycle

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sender-workers` | Number of sender threads | 1 |
| `--receiver-workers` | Number of receiver threads | 1 |
| `--msgs-per-ep` | Total messages per sender | 1000 |
| `--sender-ep-cycles` | Sender endpoint recycle count | 1 |
| `--receiver-ep-cycles` | Receiver endpoint recycle count | 1 |
| `--remove-av` | Remove old AV if AV update received | off |
| `--shared-av` | Use shared address vector | off |
| `--shared-cq` | Use shared completion queue | off |
| `--op-type` | Operation type (untagged/tagged/writedata) | untagged |
| `--random-seed` | Seed for random behavior | time(NULL) |

## Success Criteria

The test passes if:

- No endpoint setup/teardown failures
- No send/recv/write operations failures

*Completion errors and misses do not fail the test and might be expected in some scenarios.*

## Typical Use Cases

- Validate endpoint lifecycle management under stress
- Test AV update correctness during active communication
- Verify thread-safety of shared resources
- Stress-test completion queue handling with high concurrency
- Validate RMA operations with dynamic endpoint changes
