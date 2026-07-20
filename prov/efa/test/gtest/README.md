# GoogleTest-based EFA unit tests

## How to run

To run efa unit tests, you will need to have [GoogleTest 1.17](https://github.com/google/googletest/releases/tag/v1.17.0) installed.
See installation instructions [here](https://google.github.io/googletest/quickstart-cmake.html).

You will need to configure libfabric with `--enable-efa-gtest=<path_to_gtest_install>`.

An example build and run command would look like:

```bash
./autogen.sh && ./configure --enable-efa-gtest=/home/ec2-user/googletest/install
make /prov/efa/test/gtest/efa_gtest -j
```

You can then directly execute the test executable:
```
./prov/efa/test/gtest/efa_gtest
```

## File Structure
* `efa_gtest_main.cc`: Test entry point. Its `main` calls `InitGoogleMock` and `RUN_ALL_TESTS()`. Tests register themselves through the `TEST`/`TEST_F` macros, so — unlike cmocka — there is nothing to declare or add to `main` by hand.
* `efa_gtest_common_resource.{cc,h}`: Defines `struct efa_resource` and `efa_test_resource_construct`/`efa_test_resource_destruct`, which build and tear down a full set of OFI objects (fabric, domain, endpoint, EQ, AV, CQ).
* `efa_gtest_common_mocks.{cc,h}`: The `MockEfa` gmock class for libibverbs/efadv calls (e.g. `ibv_create_ah`), plus the `__wrap_*` trampolines that route those calls to the installed mock.
* `efa_gtest_common_helpers.{c,h}`: C-linkage helpers. EFA internal headers (`efa.h`, `efa_av.h`, ...) transitively pull in `unix/osd.h`, which uses C `_Complex` types that don't compile under C++. Anything a test needs from EFA internals is wrapped by a C function here and declared `extern "C"` in the header.
* `efa_gtest_{component}.cc`: The tests themselves, one file per component (e.g. `efa_gtest_conn.cc`).

## What Should be Tested
1. We make a conscious trade-off to test larger rather than smaller units. Hitting small but trivial units can increase coverage but don't test anything interesting.
2. We are biased toward testing edge cases over "happy cases", especially if the code path under test cannot be covered by integration tests.

## How to write
1. Read the [GoogleTest documentation](https://google.github.io/googletest/), particularly the [primer](https://google.github.io/googletest/primer.html), and the [gMock Cookbook](https://google.github.io/googletest/gmock_cook_book.html).
2. Pick the component file `efa_gtest_{component}.cc` (create one if needed and add it to `nodist_..._SOURCES` in `prov/efa/Makefile.include`).
3. Write a fixture class deriving from `::testing::Test`. Embed a `struct efa_resource` and any mocks (e.g. `MockEfa`); call `efa_test_resource_construct` to set up and `efa_test_resource_destruct` + `MockEfa::set(nullptr)` to tear down. See `EfaConnTest` in `efa_gtest_conn.cc`.
4. Write tests with `TEST_F(Fixture, name)`. There is no header to update and no group to register — gtest discovers tests automatically.
5. If a test needs EFA internals, expose them through a helper in `efa_gtest_common_helpers.c` rather than including the EFA headers from C++.

## Mocking
We intercept functions with the GNU linker's `--wrap` and back them with gmock for expressive expectations. The `EFA_MOCK_FUNCTIONS` X macro in `efa_gtest_common_mocks.h` is the single source of truth — it generates `MOCK_METHOD` declarations, `__real_` extern declarations, `__wrap_` trampolines, and a `MockEfa::Fn` enumerator per function automatically.

A `--wrap=<fn>` is global: it rewires *every* call to `<fn>` across the whole test binary. So the moment one test file adds a symbol to the wrap set, an installed mock would intercept that symbol everywhere — including in tests that never cared about it. To contain this, each `MockEfa` instance carries an *armed* bitset and the trampoline forwards to the mock **only for armed functions**, falling through to `__real_<fn>` otherwise. A function is automatically armed through `EFA_EXPECT_CALL` (below).

### Adding a new mock

1. Add a row `X(return_type, fn, (param decls), (arg names))` to the `EFA_MOCK_FUNCTIONS` list in `efa_gtest_common_mocks.h`. The parenthesized param/arg groups are single macro arguments (the parens shield their commas); the generators drop them into `MOCK_METHOD`, the `__real_`/`__wrap_` prototypes, and the arm enumerator.
2. Add any needed forward struct declarations at the top of the header.
3. Add `-Wl,--wrap=<fn>` to `prov_efa_test_gtest_efa_gtest_LDFLAGS` in `prov/efa/Makefile.include`.

Because the trampoline only intercepts *armed* functions, adding a new mock does not silently reroute an existing test's real calls into gMock's defaults — an unarmed function stays real. If a wrapped function does need real behavior *within* a test that also mocks it (e.g. you arm it but want some calls to run for real), route it explicitly:
```
EFA_EXPECT_CALL(mock_efa, ibv_create_ah)
        .WillRepeatedly(Invoke(__real_ibv_create_ah));
```

### Using mocks in tests

Install the mock with `MockEfa::set(&mock)` and set up expectations with **`EFA_EXPECT_CALL(mock, ...)`** — not the bare `EXPECT_CALL` — e.g. `.WillOnce(testing::Return(...))`. `EFA_EXPECT_CALL` arms the function and opens the expectation in one step; the chained action builders (`.WillOnce`/`.Times`/etc.) work exactly as on `EXPECT_CALL`. A function you never `EFA_EXPECT_CALL` stays unarmed, so its wrapped calls fall through to `__real_<fn>`.

`EFA_EXPECT_CALL(mock, fn)` matches any arguments; to match on arguments pass them as trailing macro args — `EFA_EXPECT_CALL(mock_efa, ibv_destroy_ah, &dummy_ah)` expands to `EXPECT_CALL(mock_efa, ibv_destroy_ah(&dummy_ah))`. (Matchers cannot be written as `fn(...)` inside the macro because the arming step pastes `FN_##fn`, which requires `fn` to be a bare identifier; the macro uses `__VA_OPT__` to add the matcher parentheses only when trailing args are present.)

Always clear the mock in the fixture's `TearDown` with `MockEfa::set(nullptr)` so it doesn't leak between tests. Note the arming bitset lives on the per-test `MockEfa` instance, so it resets automatically each test — but the `--wrap` symbol and the installed-mock pointer are process-global, hence the explicit clear.

### Strictness and explicit actions

Declare the mock member as `testing::StrictMock<MockEfa>`, not bare `MockEfa`. A bare mock only prints a warning when an *armed* function fires with no matching expectation; `StrictMock` makes it a hard failure. That is the behavior we want here: once you arm a function you have declared it significant, so every armed call that fires should be accounted for. Reserve `testing::NiceMock<MockEfa>` for the rare, commented case where a helper fires a high-volume call you deliberately don't want to enumerate.

Strictness is a separate concern from the *return value* of an expected call: `StrictMock` only governs whether a call was expected, never what it returns. A `StrictMock` whose expectation omits an action still returns the gMock default (0/false/NULL/default-constructed). So always give every `EFA_EXPECT_CALL` an explicit action (`WillOnce`/`WillRepeatedly`/`Times(0)`) rather than relying on that implicit default.

Because arming is opt-in, a wrapped call made during teardown falls through to `__real_<fn>` **unless the test armed that function** — so you generally do *not* need to expect the endpoint's teardown calls (e.g. the RDM CQ drain's `efa_ibv_cq_start_poll`). The exception is a function the test *does* arm that also fires during teardown: e.g. a test that arms `ibv_destroy_ah` for a peer AH must clear the mock (`MockEfa::set(nullptr)`) *before* `efa_test_resource_destruct`, or the real destroy of `self_ah` will route into the mock as an unexpected call. See `EfaConnTest`.
