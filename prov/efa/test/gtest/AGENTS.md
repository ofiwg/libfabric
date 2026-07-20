# AGENTS.md — EFA provider work

Companion to [prov/efa/test/gtest/README.md](prov/efa/test/gtest/README.md).

## Ground rules (all skills)

- **For the public API contract, read the man pages ([man/](man/)), don't
  reverse-engineer it from the code.** The `man/*.md` sources (e.g.
  [man/fi_cq.3.md](man/fi_cq.3.md), [man/fi_rma.3.md](man/fi_rma.3.md)) are the
  authoritative spec for what an `fi_*` call promises — return values, flags,
  which output fields are populated, error semantics. Deducing the contract from
  the implementation is slow and unsafe: the code may be *buggy*, and a test that
  encodes the buggy behavior as "correct" is worse than no test. Use the man page
  to decide what to assert; use the code only to trace which branch runs.
- **Assume you can't build or run locally.** The EFA provider and its gtest
  harness are Linux-only (libibverbs/efadv, GNU `ld --wrap`) and need a real EFA
  device, so a build/run is often unavailable. Self-review instead: re-read the
  changed code in full, trace the
  affected paths, and check mock cardinalities against the real call sites.
- **Don't add comments or documentation to unit tests by default.** It is the
  user's responsibility to read the code and tests and add comments where they
  judge them appropriate. After writing a test, prompt the human to review it and
  add comments where they deem necessary.
- **Keep this file current.** When a *verified* change to the gtest framework or
  EFA codebase (one the user confirmed built and passed, per the no-local-build
  rule) makes an instruction here wrong or misleading, update that instruction in
  the same change. Fix the stale contract, don't append a changelog — and only
  after the change is confirmed, never on a speculative or unlanded edit.

---

## Skill 1 — Writing a gtest unit test

File `prov/efa/test/gtest/efa_gtest_<subsystem>.cc`; fixture
`Efa<Subsystem>Test : testing::Test`; test `TEST_F(Fixture, snake_case_name)`.
`SetUp` constructs resources + `MockEfa::set(&mock)`; `TearDown`
`MockEfa::set(nullptr)` + destruct.

**When in doubt about a gtest/gmock API or behavior, consult the GoogleTest
documentation** (primer + gMock Cookbook, linked in the gtest README) — it is the
authority, not the existing tests. The current suite exercises only a slice of the
framework and is not guaranteed to model best practice, so don't infer an API's
contract or copy a pattern from a nearby test when the docs can settle it.

### Workflow

1. Pick/create the component file; if new, add it to `nodist_..._SOURCES` in
   [prov/efa/Makefile.include](prov/efa/Makefile.include).
2. Fixture derives from `::testing::Test`, embeds a `struct efa_resource` and any
   mocks. Call `efa_test_resource_construct` in SetUp, `efa_test_resource_destruct`
   in TearDown. Model on `EfaConnTest` in `efa_gtest_conn.cc`.
3. Route to the intended unit. The resource's fabric name / EP type / CQ format /
   caps decide which implementation you land in — e.g. `fi_cq_readfrom` reaches
   `efa_rdm_cq_readfrom` on the RDM fabric but base `efa_cq_readfrom` only on
   **efa-direct**. Confirm the dispatch before writing assertions.
4. To run one body over several inputs, parameterize instead of copy-pasting:
   fixture derives from `testing::TestWithParam<T>`, read the input with
   `GetParam()` in a `TEST_P(Fixture, name)`, register the set once with
   `INSTANTIATE_TEST_SUITE_P(, Fixture, <generator>, <name-fn>)` (generator e.g.
   `testing::Bool()` / `Values(...)`, optional trailing lambda names each
   instance). Model on `EfaRtmTest` in `efa_gtest_rdm_pke_rtm.cc`. Assert the
   field that varies with the parameter (Assertion strength below).

### Mocking

- Mock via the `EFA_MOCK_FUNCTIONS` X-macro + `-Wl,--wrap`. To add one: add a row
  `X(ret, fn, (param decls), (arg names))`, add forward struct decls, add
  `-Wl,--wrap=<fn>` to the LDFLAGS in Makefile.include. One row generates the
  `MOCK_METHOD`, the `__real_`/`__wrap_` prototypes, and the `MockEfa::Fn`
  enumerator that indexes the per-instance armed bitset — no extra step.
- **Expect via `EFA_EXPECT_CALL(mock, fn, ...)`, not bare `EXPECT_CALL`.** It
  arms the function *and* opens the expectation; the trampoline routes a wrapped
  call into the mock **only for armed functions** and otherwise falls through to
  `__real_<fn>`. Arming is what confines a `--wrap` (a process-wide symbol) to
  the test that cares. No-arg form matches any args; trailing args are matchers
  (`EFA_EXPECT_CALL(mock, ibv_destroy_ah, &ah)` → `ibv_destroy_ah(&ah)`).
- **Any efa provider function (not just libibverbs/efadv) can be wrapped — so
  choose the seam deliberately.** Arming means an unarmed wrapped symbol stays
  real, but a seam close to the unit under test still keeps error injection
  precise: prefer a function whose only relevant caller is the path you are
  testing (e.g. `ofi_mr_map_insert`, whose single efa caller is unambiguous)
  over a high-traffic leaf you must enumerate call-by-call once armed. The
  narrowest seam that fails the exact branch you want is the right one.
- **`StrictMock<MockEfa>` always.** Once you arm a function you have declared it
  significant, so every armed call that fires must have a matching expectation —
  an unexpected one must fail the test. `NiceMock` only as a commented,
  deliberate exception.
- **Explicit action on every `EFA_EXPECT_CALL`** — `WillOnce`/`WillRepeatedly`/
  `Times(0)`. `StrictMock` governs *whether* a call was expected, never *what it
  returns*; the implicit default (0/NULL) is a trap.
- **Arm-to-all-real is pointless.** Arming a function only to route *every* call
  to `__real_<fn>` behaves exactly like leaving it unarmed. Arming pays off when
  you intercept *some* calls and let the rest run real — sequence per-call
  `WillOnce`s with a `WillRepeatedly(Invoke(__real_<fn>))` tail, e.g. fail the
  first `ibv_create_ah` (`WillOnce(Return(nullptr))`) and let later ones succeed
  for real.
- **Explicit cardinality.** Trace the path and use `WillOnce` / `Times(N)` /
  `Times(0)` rather than a bare `WillRepeatedly` when the count is known.
- **Teardown calls fall through to real unless armed.** Because arming is
  opt-in, you generally do *not* enumerate the endpoint's destruct calls (e.g.
  the RDM CQ drain's `efa_ibv_cq_start_poll`) — they reach `__real_`. The
  exception is a function the test *arms* that also fires in destruct: e.g. a
  test arming `ibv_destroy_ah` for a peer AH must `MockEfa::set(nullptr)` before
  `efa_test_resource_destruct`, or `self_ah`'s real destroy routes into the mock
  as an unexpected call (see `EfaConnTest`).
- **Static-inline functions** (`efa_qp_post_*`, `efa_ibv_cq_*` in
  `efa_data_path_ops.h`) are only linkable under `#if EFA_UNIT_TEST`. Mocking them
  requires configuring with **both** `--enable-efa-unit-test` AND
  `--enable-efa-gtest`; gtest-only silently binds `--wrap` to nothing → false pass.

### C/C++ bridge

- EFA internal headers won't compile in a `.cc` for two independent reasons, both
  reached transitively from `efa.h`: (1) `unix/osd.h` includes `<complex.h>` and
  uses C `_Complex` types, and (2) `ofi.h` → `ofi_atom.h` includes C11
  `<stdatomic.h>` (`atomic_int`/`_Atomic`/`atomic_*`), which is not C++-compatible
  — C++ has its own `<atomic>`, and the `extern "C"` guard doesn't help because
  these are language-level type/keyword incompatibilities, not linkage. Access
  internals through C helpers in `efa_gtest_common_helpers.c` (`extern "C"`).
  Prefer forward-declaring a production function over a passthrough wrapper; write
  a wrapper only to reach opaque struct internals.
- **Put test-specific C helpers in a per-component `.c/.h` pair, not
  `common_helpers`.** When a test's own logic must live in C (because it touches
  EFA internals per above), give that component its own bridge pair — e.g.
  `efa_gtest_rdm_pke_utils.{c,h}` for `efa_gtest_rdm_pke_rtm.cc`; both files go in
  `nodist_..._SOURCES` (headers picked up via `-I`). Reserve
  `efa_gtest_common_helpers.{c,h}` for genuinely common helpers you expect
  multiple test files to reuse. These per-component helpers are effectively test
  bodies that had to move into C — keep them next to the test that owns them.
  When a helper *is* the test body (moved into C only because it touches EFA
  internals), name it `efa_test_<test_name>` after the `TEST_F`/`TEST_P` in the
  `.cc` that calls it, so the C body and its owning test are unambiguously paired.

### Assertion strength

- **Assert the produced output, not just the mock's call count.** A satisfied
  `EFA_EXPECT_CALL` proves a value was *read*, not that it *landed* in the
  completion/error entry, so don't stop at call count — also check `op_context`,
  `flags`, `len`, `data`, `prov_errno`, `err`, `src_addr`, etc. Call count is
  still worth asserting when it *is* the contract (a cardinality like `Times(2)`,
  or a `Times(0)` proving a branch was skipped); assert both when both matter.
- **A suppressed output is a claim — assert it** (read back `-FI_EAGAIN` when a
  branch should stage nothing). A comment is not a test.
- **Use distinct sentinels to prove which branch ran** (e.g. `op_context == &ctx`
  where `&ctx != direct_ope`), not a pointer both branches share.
- **Parameterized tests assert the per-parameter difference** (the field that
  varies by opcode). Fold opcode-only variants into `TEST_P`.
- **Read back without re-triggering the path.** `fi_cq_read`/`readfrom` call
  `progress()` and re-enter the mocked poll loop, breaking `StrictMock`; use
  `ofi_cq_read_entries` (no progress) or `fi_cq_readerr` (no progress).
- **Stop before brittle.** Don't assert log strings, exact error text, or values
  the contract doesn't promise; prefer `EXPECT_NE(x, 0)` when the contract only
  guarantees "nonzero/derived".

### Host-capability gating (skip, don't fake)

- Paths needing real hardware (efa-direct FI_RMA read/write need Nitro v4+ RDMA)
  cannot be faked — forcing caps gets past `fi_getinfo` but `fi_enable` still
  creates a real QP via `efadv_create_qp_ex` and the device rejects it
  (`-FI_EOPNOTSUPP`). Query the real capability in `SetUp()` (C helper wrapping
  `efa_device_support_rdma_read()/_write()`) and `GTEST_SKIP()` when absent.
  Skipping in SetUp means the body never runs (also dodges the fatal-ASSERT-in-
  helper fall-through trap). Wrap `construct()` in `ASSERT_NO_FATAL_FAILURE`.

### State hygiene

- Save any global mutable state (`efa_env` fields like `track_mr`) in SetUp,
  restore in TearDown, so tests stay order-independent. Prefer instance-local
  seams (e.g. swap `rdm_domain->shm_domain` around the call) over mutating globals.
- Pair every alloc/free; no leaks, no double-frees.

---

## Skill 2 — Reviewing a gtest unit test

Work through these when reviewing a new/modified `efa_gtest_*.cc`. Items 5–7 are
highest-value — they catch what a passing run does not (wrong function under test,
mismatched cardinality, a branch never hit).

1. **Conformance** — follows the gtest README and this file (mock style,
   strictness, C/C++ bridge, static-inline mocking, naming).
2. **Comments** — terse, contract-focused; flag any that restate code; prefer
   function-name references over `file.c:812` line refs (they rot).
3. **No logical overlap; parameterize near-duplicates** — each test should hit a
   distinct branch. If two `TEST_F`s share the same body and differ only in a set
   of values (opcode, CQ format, flag combo, expected field) that could be a
   `GetParam()` field, they're one `TEST_P` over a case table, not copy-paste.
   The test asserts the per-parameter difference (item 10); if it can't — i.e. the
   variants really do exercise different logic/branches — keep them separate.
4. **Naming** — fixture/test names describe the behavior, not overly long.
5. **Mock expectations match the real path** — re-trace production for every test;
   confirm each cardinality and action, *including teardown calls*. The *absence*
   of an expectation is a claim (no `slid`/`src_qp` ⇒ non-`FI_SOURCE` branch) —
   verify those too.
6. **Each fixture routes to the intended unit** — confirm fabric name / EP type /
   CQ format / caps dispatch into the function the test means to cover. A test that
   passes against the wrong implementation is a trap.
7. **Strictness and teardown** — `StrictMock` everywhere (no bare/`Nice` without a
   commented reason), expectations via `EFA_EXPECT_CALL` (never bare
   `EXPECT_CALL`), `MockEfa::set(nullptr)` in every TearDown. Any function the
   test arms that also fires during destruct is either uninstalled before
   destruct or routed to `__real_`.
8. **Resource/global-state hygiene** — alloc/free pairing; global state saved in
   SetUp and restored in TearDown.
9.  **Assertion strength** — asserts the observable contract (output fields set,
    output suppressed, per-parameter difference), proves *which* branch ran via
    sentinels, stops short of brittle log/text assertions. Flag any test that
    checks only the return code or a satisfied `EFA_EXPECT_CALL` while leaving an
    output field unverified.
10. **Build wiring** — new mock has a matching `-Wl,--wrap=<fn>` in
    Makefile.include; new file added to `nodist_..._SOURCES`.
11. **No pointless arming** — flag any `EFA_EXPECT_CALL` whose every action routes
    to `__real_<fn>` (a lone `WillRepeatedly(Invoke(__real_...))`): arming only to
    run fully real behaves exactly like not arming, so either drop the expectation
    or, if the intent was partial interception, sequence the real `WillOnce`s
    before the `__real_` tail.

---

## Skill 3 — Reviewing an EFA source file for bugs

Read the file in full and trace each code path by hand (no build here). The
bug classes below are the ones this codebase has actually shipped and fixed —
prioritize them.

### High-value bug patterns

- **Error code clobbered by cleanup.** A failure path calls a cleanup function and
  writes its (successful) return over the real error: `ret = efa_rdm_mr_dereg_impl(...)`
  turned a failed registration into a success that returned a deregistered MR. Fix
  is usually to drop the `ret =`. Check every `goto err` / cleanup block: does it
  preserve the original `ret`?
- **Buffer bounds skipped on a bypass path.** The util layer bounds a copy with
  `MIN(user_size, prov_size)`; a provider "bypass" path (e.g. efa-direct
  `efa_cq_readerr` → `efa_write_error_msg`) can `snprintf` a fixed max into the
  caller's smaller buffer. For every write into a caller-supplied buffer, confirm
  the caller's declared capacity actually bounds it — and that the reported size is
  truncation-aware, not a hardcoded max.
- **Use-after-free / free-then-use across a boundary.** Completion handlers free
  the `txe`/`ope` at `bytes_acked == total_len`; anything read after that boundary
  must come from an object that outlives it (peer/domain counters), not the freed
  one. Check clone/release paths (cf. the `rdm_pke_release_cloned` UAF fix).
- **Double-free / missing-free on error paths.** Pair every alloc with exactly one
  free across all branches; a partially-constructed object freed by both the error
  path and the destructor is a double-free.
- **Return-value defaults masking failure.** A function that returns 0 on an
  untaken path, or an uninitialized `ret`, can report success spuriously.
- **Refcount asymmetry on error paths.** An `*_open`/`bind` that increments a
  refcount (util_domain, av, eq, cntr) on success but whose error path returns
  without the matching decrement leaks the reference — cf. the `rdm_cntr_open`
  util_domain leak. For every refcount bump, check that *every* exit after it
  (each `goto err`, early return) either owns the ref or drops it; close/unbind
  must balance open/bind exactly.
- **Locking: contention and data races.** Check for lock-ordering and
  double-locking bugs — a path that re-acquires a lock it already holds (or an
  error/cleanup branch that unlocks a lock it never took), and two locks taken in
  opposite orders on different paths (deadlock). When the object is reached
  concurrently (progress thread vs. app thread, CQ poll vs. completion), confirm
  every access to shared state is under the intended lock — an unguarded read/write
  or a field mutated outside the lock that protects its invariant is a race.

### Method

- **Judge public-facing behavior against the man page; judge internal details
  for self-consistency.** Where a behavior is part of the public API contract,
  verify it against [man/](man/) (per the ground rule) rather than the code, so
  you neither flag intended behavior nor bless a buggy implementation. But the
  man pages don't cover everything — plenty of implementation details have no
  documented contract; for those, the bar is that the code is internally bug-free
  and self-consistent (no UAF, no double-free, no path contradicting another).
- **Identify which fabric/EP the function serves.** Base vs. RDM vs. efa-direct
  paths differ: a bug may exist only on the bypass path (efa-direct `efa_cq_*`)
  while the RDM path routes through safe util code. State which path you're
  analyzing.
- **Trace each error/early-return branch to its observable effect** — what does the
  caller see in the return code and in every output field? Mismatches there are the
  bugs unit tests are meant to pin.
- **Check capability gating.** Code assuming RDMA read/write, HMEM/CUDA, or shm may
  be reached on a device that lacks it; confirm the guard exists and is correct.
- **When you find a bug, describe the failing input → wrong output concretely,** and
  (if asked to test it) design a *failing* unit test first using the seams in
  Skill 1 — pick a `--wrap` seam whose only caller is unambiguous so the
  fault-injection is targeted (e.g. `ofi_mr_map_insert` has exactly one efa caller).
