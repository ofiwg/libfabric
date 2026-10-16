# EFA RDM address vector: data structures and locking

This document describes how the EFA RDM provider resolves a remote endpoint to
its per-endpoint peer state, which locks protect each piece of that state, and
why the scheme is free of the races that the previous design had.

## 1. Why the previous design kept producing races

A single fact -- "remote endpoint R is represented on local endpoint E by peer
P" -- used to be spread over four independently published maps:

| map | key | owner | written under |
| --- | --- | --- | --- |
| `cur_reverse_av` | (AHN, QPN) | AV | `util_av.lock` |
| `cur_reverse_av_implicit` | (AHN, QPN) | AV | `util_av_implicit.lock` |
| `ep->fi_addr_to_peer_map` | explicit fi_addr | EP | `ep->ctrl_lock` |
| `ep->fi_addr_to_peer_map_implicit` | implicit fi_addr | EP | `ep->ctrl_lock` |

The implicit and explicit AV entries were also two different objects. Promoting
a peer from the implicit to the explicit AV therefore had to (1) allocate a new
AV entry, (2) re-key every endpoint's peer from the implicit fi_addr to the
explicit fi_addr, (3) rewrite `peer->av_entry`, and (4) publish the new entry in
the reverse AV and in `addr_to_entry_map`, while freeing the implicit entry.

The readers combine two lock-free lookups (reverse AV, then peer map), and the
peer creators (TX path, CQ read path, AV insert path) each decide "does a peer
already exist?" by looking in whichever map *they* key by. Any interleaving that
lets a reader or a creator observe the middle of the re-key is a bug:

* **Publish order bug (fixed in 4e4572b5).** The promotion published the entry
  in `cur_reverse_av` before re-keying the peer maps. A CQ reader found the new
  fi_addr, missed in `fi_addr_to_peer_map`, and created a second peer.
* **TX path bug.** The promotion makes the entry visible in
  `addr_to_entry_map[X]` before it has moved the existing peer into
  `fi_addr_to_peer_map[X]`. `efa_rdm_ep_get_peer_explicit()` treats
  "`addr_to_entry_map` hit + peer map miss" as "the peer does not exist yet"
  and creates one under `ctrl_lock`; the AV insert path then moves the old
  implicit peer into the same slot. The creator and the mover each hold a
  lock, but they decide existence by looking at *different* maps, so the
  re-check under `ctrl_lock` cannot see the other's work. Two peers end up
  owning one remote endpoint (two `next_msg_id` streams, two reorder buffers).

Fixing the ordering one interleaving at a time does not converge: every new
reader path (SHM lookup, RMA, handshake responses, `foreach_unspec_addr`) adds
another pair of maps that must be published in some order relative to the
others. The redesign removes the re-key instead.

## 2. Design in one paragraph

Every remote endpoint known to an AV is represented by exactly one **conn**
object (`struct efa_rdm_av_entry`). The conn is allocated when the remote first
enters the AV -- either explicitly through `fi_av_insert()` or implicitly when a
packet arrives from an unknown sender -- and it keeps the same address and the
same **peer index** until it leaves the AV. Promotion from the implicit to the
explicit AV changes only which maps point *at* the conn; it never creates,
moves, or rewrites a peer. Each endpoint keeps a single `peer_map`, indexed by
the conn's immutable peer index, and every path that needs a peer calls one
function, `efa_rdm_ep_get_peer()`, which looks the peer up lock-free and, on a
miss, creates it under `ep->ctrl_lock` after re-checking the same slot. Because
there is exactly one key per remote endpoint per local endpoint, and exactly one
place where peers are created, a duplicate peer cannot be constructed by any
interleaving.

## 3. Data structures

### 3.1 AV level (`struct efa_rdm_av`)

```
        explicit util_av                          implicit util_av
        (raw addr -> fi_addr X)                   (raw addr -> fi_addr I)

 addr_to_entry_map[X] ---> +---------------------------+ <--- addr_to_entry_map_implicit[I]
 cur_reverse_av[ahn,qpn]-> | conn (efa_rdm_av_entry)   | <--- cur_reverse_av_implicit[ahn,qpn]
 prv_reverse_av{..,qkey}-> |  ep_addr, ah, fi_addr,    | <--- prv_reverse_av_implicit{..,qkey}
                           |  implicit_fi_addr,        |
                           |  shm_fi_addr, peer_idx,   |
                           |  released                 |
                           +---------------------------+
                              allocated from conn_pool (indexed ofi_bufpool)
```

* **`conn_pool`** -- an indexed `ofi_bufpool` that owns every conn. A conn's
  peer index is its buffer index (`ofi_buf_index()`), so indexes are dense and
  are reused only after the conn is freed. Pool memory is never returned to the
  OS while the AV is open, so a stale lock-free pointer to a freed conn reads
  valid (if recycled) memory rather than faulting. The pool has its own leaf
  lock, `conn_pool_lock`, held only around `ofi_ibuf_alloc()`/`ofi_ibuf_free()`:
  explicit conns are created under `util_av.lock`, implicit ones under
  `util_av_implicit.lock`, and AH eviction frees implicit conns of any AV in the
  domain while holding only that AV's implicit lock.
* **util AV entries carry no context.** Each `util_av` (explicit and implicit)
  still provides the raw-address hash and hands out fi_addrs; FI_AV_TABLE
  semantics (the first `fi_av_insert()` returns 0, each later one the next
  integer) come from the explicit `util_av` exactly as before. A raw-address
  lookup yields an fi_addr, and the fi_addr is mapped to the conn through
  `addr_to_entry_map` or `addr_to_entry_map_implicit`, which point directly at
  the conn. There is no pointer from the util AV entry to the conn, so
  `addr_to_entry_map` lookups cost the same number of hops as before.
* **`addr_to_entry_map`** (explicit fi_addr -> conn) and
  **`addr_to_entry_map_implicit`** (implicit fi_addr -> conn): `efa_av_array`s,
  lock-free reads. `efa_av_array` publishes every slot, chunk and chunk-table
  pointer with a release store and reads it with an acquire load. (It used to
  use standalone fences around plain accesses, which is a data race under the
  C11 model and which ThreadSanitizer reports on every concurrent lookup.)
* **`cur_reverse_av`** and **`cur_reverse_av_implicit`** ((AHN, QPN) -> conn):
  `efa_av_array`s, lock-free reads. A conn is in exactly one of them, except
  for the instant during promotion described in section 5.4, when it is briefly
  in both; both resolve to the same conn, so that is harmless.
* **`prv_reverse_av`** and **`prv_reverse_av_implicit`** ((AHN, QPN, connid) ->
  conn): uthash maps for conns displaced by QPN reuse. Locked reads only.

### 3.2 Conn (`struct efa_rdm_av_entry`)

| field | written | read | protection |
| --- | --- | --- | --- |
| `efa_av_entry.ep_addr`, `.ah` | at conn creation | anywhere | immutable while published |
| `peer_idx` | at conn creation | anywhere | immutable |
| `shm_fi_addr` | at explicit creation, before publication | TX path, lock free | immutable while published |
| `efa_av_entry.fi_addr` | creation, promotion | data path (completion `src_addr`, SRX matching, logging) | atomic load / store |
| `implicit_fi_addr` | creation, promotion, release | AV paths only | `util_av_implicit.lock` |
| `implicit_av_lru_entry` | implicit AV paths | implicit AV paths | `util_av_implicit.lock` |
| `ah_implicit_conn_list_entry` | AH bookkeeping | AH bookkeeping | `util_domain.lock` |
| `released` | release | peer creation slow path | atomic; see 5.7 |

### 3.3 Endpoint level (`struct efa_rdm_ep`)

* **`peer_map`** -- a single `efa_av_array` indexed by `conn->peer_idx`. It
  replaces `fi_addr_to_peer_map` and `fi_addr_to_peer_map_implicit`. Reads are
  lock free; every insert and remove holds `ep->ctrl_lock`.
* **`efa_rdm_peer_pool`, `peer_robuf_pool`** -- unsynchronized `ofi_bufpool`s
  used only for peer construction and destruction. Guarded by `ep->ctrl_lock`.
* **`tx_peer_cache`** -- an `efa_av_array` indexed by explicit fi_addr, holding
  copies of `peer_map` entries so the TX path reaches its peer in one lookup.
  It is only a cache: it never decides whether a peer exists. Entries are copied
  from `peer_map` by the TX slow path and cleared when the peer is destroyed,
  both under `ep->ctrl_lock`. Reads are lock free.
* **`peer->av_entry`** points at the conn and never changes for the life of the
  peer.

## 4. Locks

| lock | protects | type (SAFE / COMPLETION / DOMAIN) |
| --- | --- | --- |
| `ep->srx_lock` | per-EP protocol state (unchanged; see the data-path lock summary) | MUTEX / NOOP / NOOP |
| `util_domain.lock` | AH map and refcounts, AH LRU, `implicit_conn_list`, `mr_map` | MUTEX / MUTEX / NOOP* |
| `util_av.lock` | explicit util AV hash and fi_addr allocator, every write to `cur_reverse_av`, `addr_to_entry_map`, `prv_reverse_av`; `shm_used` | MUTEX / MUTEX / NOOP* |
| `util_av_implicit.lock` | implicit util AV hash and allocator, every write to `cur_reverse_av_implicit` and `addr_to_entry_map_implicit`, `prv_reverse_av_implicit`, LRU list, evicted-peer set, `conn->implicit_fi_addr` | MUTEX / MUTEX / NOOP* |
| `util_av.ep_list_lock` | the list of endpoints bound to the AV | MUTEX / NOOP* / NOOP* |
| `ep->ctrl_lock` | every write to `ep->peer_map` and `ep->tx_peer_cache`; `efa_rdm_peer_pool`; `peer_robuf_pool` | MUTEX / MUTEX / NOOP |
| `conn_pool_lock` | `conn_pool` | MUTEX / MUTEX / NOOP* |

\* NOOP only with `FI_PROGRESS_CONTROL_UNIFIED`.

**Lock order** (outermost first), declared to Clang in
`efa_thread_annotations.h` and checked at compile time:

```
srx_lock -> util_domain.lock -> util_av.lock -> util_av_implicit.lock
         -> util_av.ep_list_lock -> ctrl_lock
```

`ctrl_lock` and `conn_pool_lock` are leaves: nothing is acquired while either
is held. Inserting into the implicit AV additionally requires `util_av.lock`,
because whether an address is "in neither AV" can only be decided while both AVs
are stable (section 5.3); `efa_rdm_av_insert_one_implicit()` declares that
requirement to the analysis.

## 5. Operations

### 5.1 Peer lookup / creation -- `efa_rdm_ep_get_peer(ep, conn)`

```
peer = peer_map[conn->peer_idx]            // lock free
if (peer) return peer;                     // fast path
lock(ep->ctrl_lock)
peer = peer_map[conn->peer_idx]            // re-check the SAME slot
if (!peer && !conn->released) {
        peer = construct(ep, conn)
        peer_map[conn->peer_idx] = peer    // publish (efa_av_array wmb)
}
unlock(ep->ctrl_lock)
```

This is the only place a peer is created. The "does a peer exist?" decision is
made by re-reading the slot that every creator and every remover writes, under
the lock that every creator and every remover holds. That is the property the
previous design lacked.

### 5.2 TX path (`fi_send`, `fi_read`, `fi_write`, atomics)

```
peer = tx_peer_cache[X]                    // lock free, the common case
if (peer) return peer;
conn = addr_to_entry_map[X]                // lock free
peer = efa_rdm_ep_get_peer(ep, conn)       // creates the peer if needed
lock(ep->ctrl_lock)
if (!conn->released && peer_map[conn->peer_idx] == peer)
        tx_peer_cache[X] = peer            // fill the cache
unlock(ep->ctrl_lock)
```

No AV lock is taken. The libfabric spec requires `fi_av_insert()` to have
returned X before X is used, and once it has, `addr_to_entry_map[X]` holds the
conn and the conn's `peer_idx` never changes, so the TX path cannot observe a
half-done insert or promotion.

The cache keeps the TX path at the old design's single lookup. Without it, the
TX path would read `addr_to_entry_map[X]` and the conn before it could index
`peer_map`, one more dependent load than before. The cache is safe because it
only ever holds a copy of the `peer_map` slot that decided the peer exists:

* it is filled under `ctrl_lock`, and only while that slot still holds the same
  peer and the conn is not released;
* removal clears `tx_peer_cache[X]` together with the `peer_map` slot, under the
  same lock, before X is freed and can be handed to another address (5.7);
* promotion does not need to touch it: an implicit conn has no explicit fi_addr,
  so it has no cache slot until the TX path first sends to X after
  `fi_av_insert()` has returned it.

The SHM-peer check, when SHM is enabled, still reads `conn->shm_fi_addr`
through `addr_to_entry_map[X]` before the TX path proper, as before.

Dependent loads on the fast paths, old design versus this one:

| path | old | new |
| --- | --- | --- |
| TX | `fi_addr_to_peer_map[X]` -> peer | `tx_peer_cache[X]` -> peer |
| CQ read | `cur_reverse_av[ahn,qpn]` -> entry -> `fi_addr_to_peer_map[fi_addr]` -> peer | `cur_reverse_av[ahn,qpn]` -> conn -> `peer_map[peer_idx]` -> peer |

`addr_to_entry_map` lookups cost the same as before: the map points directly at
the conn.

### 5.3 CQ read path

Fast path, no locks:

```
conn = cur_reverse_av[ahn, qpn]            // explicit AV only
if (conn && connid matches conn->ep_addr.qkey)
        return efa_rdm_ep_get_peer(ep, conn)
```

If the packet carries the raw address but the device has not yet reported an
AHN for the sender (startup), the explicit util AV hash is searched under
`util_av.lock` only, and the resulting fi_addr is mapped to the conn through
`addr_to_entry_map`.

Slow path, taken for a miss, a stale connid, or any implicit peer: take
`util_domain.lock -> util_av.lock -> util_av_implicit.lock` and, without
releasing them, resolve the conn in this order -- explicit reverse AV (cur and
prv), explicit raw-address hash, implicit reverse AV (cur and prv), implicit
raw-address hash -- drop the packet if the sender was evicted, otherwise insert
a new implicit conn. Then call `efa_rdm_ep_get_peer()` (`ctrl_lock` nests
inside), move an implicit conn to the tail of the LRU, and drop the AV locks.

Holding the AV locks across resolve-and-create is what makes "the address is in
neither AV, insert it implicitly" atomic with respect to `fi_av_insert()`, and
it guarantees the conn cannot be released between being resolved and having its
peer created.

### 5.4 `fi_av_insert()` of an address already in the implicit AV (promotion)

Under `util_domain.lock -> util_av.lock -> util_av_implicit.lock`:

1. Allocate the explicit fi_addr X from the explicit util AV.
2. Reserve everything the commit needs: the `addr_to_entry_map[X]` slot, the
   `cur_reverse_av` slot, and a `prv_reverse_av` node if that slot is
   occupied. On failure, free X and return; nothing else has been touched.
3. Commit (cannot fail):
   1. `conn->fi_addr = X` (atomic release store);
   2. add the conn to `cur_reverse_av` (CQ fast path can now find it);
   3. `addr_to_entry_map[X] = conn` (TX path can now find it);
   4. remove the conn from `cur_reverse_av_implicit` / `prv_reverse_av_implicit`,
      clear `addr_to_entry_map_implicit[I]`, free I, clear
      `conn->implicit_fi_addr`, unlink it from the LRU and from the AH's
      implicit conn list, and move its AH reference from implicit to explicit.
4. Call `foreach_unspec_addr` on every bound endpoint's SRX, so unexpected
   messages already received from the sender are re-filed under X.

No endpoint's `peer_map`, and no peer, is read or written. Whatever peer each
endpoint already had for this remote is still at `peer_map[conn->peer_idx]` and
is found by every subsequent lookup through either the explicit or the implicit
route. This is requirement 4 ("transfer cleanly") by construction: there is
nothing to transfer. Step 3.1 precedes 3.2 so that a reader that finds the conn
in `cur_reverse_av` also sees the explicit fi_addr; the `efa_av_array` publish
barrier provides the ordering.

### 5.5 `fi_av_insert()` of a new address

Under `util_domain.lock -> util_av.lock`: allocate X and a conn, allocate the
AH, reserve the map slots, insert into SHM if the peer is local, then publish in
`cur_reverse_av` and `addr_to_entry_map[X]`. The conn is new, so no endpoint has
a peer for its `peer_idx` (section 5.7 explains why a reused index is clean), and
the order of the two publications does not matter: neither reader creates
anything except through `efa_rdm_ep_get_peer()`.

### 5.6 Implicit insert

Done only by the CQ slow path, under all three AV locks, after the resolution in
5.3 has established the address is in neither AV: allocate I and a conn,
allocate the AH (implicit reference), insert into the LRU (which may evict, see
5.8), and publish in `cur_reverse_av_implicit` and
`addr_to_entry_map_implicit[I]`.

### 5.7 `fi_av_remove()`

Under `util_domain.lock -> util_av.lock`:

1. Unpublish: clear `addr_to_entry_map[X]` and remove the conn from
   `cur_reverse_av` / `prv_reverse_av`.
2. `conn->released = true` (atomic release store).
3. Under `util_av.ep_list_lock`, for each bound endpoint, take its `ctrl_lock`,
   clear `tx_peer_cache[X]` and `peer_map[conn->peer_idx]`, and destroy the
   peer.
4. Remove from SHM, release the AH, free X, and return the conn to `conn_pool`.

A creator that read the conn lock-free before step 1 and reaches the
`ctrl_lock` re-check after step 3 has visited its endpoint sees
`released == true` (the store in step 2 happens before step 3 takes that
endpoint's `ctrl_lock`) and does not create a peer, so a peer can never be
left behind in a slot whose `peer_idx` is later reused. The TX cache fill in 5.2
makes the same `released` check under the same lock, so a cache slot cannot be
left behind for an fi_addr that is later reused either. This is best effort
only, as allowed by requirement 6: an application that removes an address while
traffic to or from it is still in flight is outside the spec.

### 5.8 Implicit AV eviction

Triggered by the LRU limit (`FI_EFA_IMPLICIT_AV_SIZE`) during an implicit insert,
or by `ibv_create_ah` running out of AHs. Same steps as 5.7, on the implicit
maps, with all three AV locks held. See section 8 for the remaining hazard.

### 5.9 Endpoint bind and close

Binding adds the endpoint to `util_av.ep_list` with an empty `peer_map`; it
cannot have peers for any conn yet. Closing unbinds the endpoint (under
`ep_list_lock`, so no AV operation is mid-iteration over it) before destroying
its peers, so no AV operation can reach a closing endpoint's `peer_map`.

### 5.10 AV close

Releases every conn by walking the two forward maps (`addr_to_entry_map` and
`addr_to_entry_map_implicit`). Every live conn is in exactly one of them, so
this also covers conns that QPN reuse moved into a `prv_reverse_av`.

## 6. How the requirements are met

1. **Lock-free fast paths.** The TX path does one lock-free read, of
   `tx_peer_cache`, up to the WQE post. The CQ fast path does one lock-free
   explicit reverse-AV read and one lock-free peer-map read. Locks are taken only
   to create a peer the first time, to fill the TX cache the first time, or for
   the implicit AV.
2. **FI_AV_TABLE numbering.** Explicit fi_addrs still come from the explicit
   `util_av` in insertion order. Implicit conns use a separate `util_av` and
   never consume an explicit fi_addr.
3. **Implicit AV.** Unchanged in behaviour: unknown senders get an implicit conn
   under the AV locks, with LRU and evicted-peer tracking.
4. **Clean implicit-to-explicit transfer.** The conn and its peers do not move
   (5.4).
5. **The TX path reads only the explicit AV, and only on its slow path.** The
   common case is a `tx_peer_cache` hit. Lazy peer creation on the TX path takes
   only `ctrl_lock`.
6. **Removal with pending operations is not defended,** but the `released`
   check (5.7) and pool-backed conns avoid leaving a stale peer behind or
   faulting, at no cost on the fast path.
7. **Peers are still created lazily.** Eager creation was considered: it would
   make `fi_av_insert()` O(bound endpoints) and endpoint enable O(AV size) for no
   locking benefit, because the single creation function in 5.1 already makes
   lazy creation race-free.

## 7. Why the two known bugs cannot recur

* **Publish order.** Promotion never touches a peer map. A CQ reader that finds
  the conn through any map reaches the same `peer_map[conn->peer_idx]` slot,
  which already holds the existing peer.
* **TX path.** The TX path, the CQ path, and the AV paths all decide whether a
  peer exists by reading the same slot under the same lock (5.1). Promotion does
  not create peers at all, so there is no second creator to race with. The TX
  cache is filled only from that slot, so it cannot introduce a second peer.

## 8. Known remaining hazards (outside this change)

* **Eviction destroys other endpoints' peers synchronously.** Eviction is
  provider-initiated, so requirement 6 does not cover it. The evicting thread
  holds the AV locks and each victim endpoint's `ctrl_lock`, but not that
  endpoint's `srx_lock` -- it cannot, since `srx_lock` is outside the AV locks
  in the lock order. Another thread that resolved the same peer earlier and is
  still processing a packet for it can therefore use the peer after it is
  destroyed. This hazard is unchanged from the previous design. It is off by
  default (`FI_EFA_IMPLICIT_AV_SIZE=0`) and otherwise only triggered when the
  device runs out of AHs. The fix is deferred reaping: eviction unpublishes the
  conn and queues it on each bound endpoint, and each endpoint destroys its own
  peer from its progress loop under its own `srx_lock`. The conn (and its AH)
  is freed when the last endpoint has reaped it.
* **`fi_av_remove()` destroys peers without `srx_lock`,** for the same lock order
  reason. Covered by requirement 6.
* **AH eviction across AVs.** When an `fi_av_insert()` on one AV runs out of AHs,
  eviction may release implicit conns that belong to a different AV of the same
  domain. It takes that AV's implicit lock, but not its `util_av.lock`, so it
  can race with that AV's CQ slow path deciding "in neither AV". Unchanged from
  the previous design; it needs several AVs on one domain and AH exhaustion.
* **`foreach_unspec_addr` ordering.** A message from a promoted sender that is
  processed after step 3.2 but before step 4 of 5.4 is filed under X ahead of
  older unexpected messages that are still on the unspecified queue. This is a
  property of the util SRX interface and is unchanged from the previous design.

## 9. Bugs found in the previous implementation along the way

* LRU eviction recorded the address being *inserted*, not the evicted one, in
  the evicted-peers set.
* AH eviction freed implicit entries without unlinking them from the implicit
  LRU list, leaving freed nodes linked into it.
* Under ThreadSanitizer the previous design reports, among others: promotion
  rewriting `peer->av_entry` while `efa_rdm_pke_sendv()`, the CQ read path and
  `efa_rdm_rxe_report_completion()` read it; and
  `efa_rdm_av_entry_alloc_implicit()` initializing an entry that a completion
  was still reading through a peer -- the implicit entry freed by an earlier
  promotion and recycled, i.e. a use-after-free. Neither can happen once peers
  point at a conn that promotion does not free.

## 10. Static and dynamic verification

* Clang Thread Safety Analysis (`--enable-thread-safety-analysis`,
  `-Werror=thread-safety`) checks the `GUARDED_BY` fields in section 3, the
  `REQUIRES` contracts of the map writers, and the lock order in section 4.
* `fi_efa_multi_ep_stress --shared-av --insert-sender-addr` exercises promotion
  concurrently with CQ reads on other endpoints.
* `fi_efa_av_insert_race` runs many threads per process. Each thread owns an
  endpoint on a shared AV, receives from remote endpoints before and after
  inserting them (implicit and explicit CQ read paths), inserts the senders it
  hears from (promotion concurrent with other threads' CQ reads), and sends to
  the same remote endpoints (TX-path lazy peer creation concurrent with
  promotion). Between rounds it removes every address, so each round starts
  from implicit entries again and reuses freed peer indexes. It checks
  exactly-once and in-order delivery per sender, fi_addr agreement between
  threads, FI_AV_TABLE numbering, `fi_av_lookup`, and the reported source
  address.
* Both fabtests run clean under ThreadSanitizer with
  `prov/efa/test/tsan.supp`, which suppresses only the deliberate
  `srx_lock`/`progress_ep_list_lock` inversion and a util-layer store to
  `util_domain.srx`. The same suppressions leave the previous design's AV data
  races reported.
