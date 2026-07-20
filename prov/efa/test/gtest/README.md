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
We intercept functions with the GNU linker's `--wrap` and back them with gmock for expressive expectations. The `EFA_MOCK_FUNCTIONS` X macro in `efa_gtest_common_mocks.h` is the single source of truth — it generates `MOCK_METHOD` declarations, `__real_` extern declarations, and `__wrap_` trampolines automatically.

### Adding a new mock

1. Add a row `X(return_type, fn, (param decls), (arg names))` to the `EFA_MOCK_FUNCTIONS` list in `efa_gtest_common_mocks.h`. The parenthesized param/arg groups are single macro arguments (the parens shield their commas); the generators drop them into `MOCK_METHOD` and the `__real_`/`__wrap_` prototypes.
2. Add any needed forward struct declarations at the top of the header.
3. Add `-Wl,--wrap=<fn>` to `prov_efa_test_gtest_efa_gtest_LDFLAGS` in `prov/efa/Makefile.include`.

If adding a mock breaks other tests, it's likely due to the new mock is being unintentionally called in a test where a real function call is expected. By default, mocked functions will return 0/false/NULL/default construct, and this may not be the expected behavior of the real functions. There are two solutions to this:

1. Define multiple mock classes analogous to `MockEfa`, to isolate the mocks, and only install the mocks you need by `MockEfa::set(&mock_efa);`
2. Set up the correct expectation to route all calls to a specific wrapped function to the real function, like
```
ON_CALL(*this, ibv_create_ah(_,:_))
        .WillByDefault(Invoke(__real_ibv_create_ah));
```
A mixture of both strategy can be used, to minimize the changes required.

### Using mocks in tests

Install the mock with `MockEfa::set(&mock)` and set up `EXPECT_CALL(mock, ...)` expectations (e.g. `.WillOnce(testing::Return(...))`). When no mock is installed, the trampoline falls through to the real `__real_<fn>` implementation.

Because `--wrap` rewires every call to the function across the whole test binary, always restore the default in the fixture's `TearDown` — clear the mock with `MockEfa::set(nullptr)` — so mocks don't leak between tests.

### Strictness and explicit actions

Declare the mock member as `testing::StrictMock<MockEfa>`, not bare `MockEfa`. A bare mock only prints a warning when a wrapped function fires with no matching `EXPECT_CALL`; `StrictMock` makes it a hard failure. That is the behavior we want here: `--wrap` intercepts the function process-wide, so every wrapped call that fires during a test is significant and should be accounted for. Reserve `testing::NiceMock<MockEfa>` for the rare, commented case where a helper fires a high-volume call you deliberately don't want to enumerate.

Strictness is a separate concern from the *return value* of an expected call: `StrictMock` only governs whether a call was expected, never what it returns. A `StrictMock` whose `EXPECT_CALL` omits an action still returns the gMock default (0/false/NULL/default-constructed). So always give every `EXPECT_CALL` an explicit action (`WillOnce`/`WillRepeatedly`/`Times(0)`) rather than relying on that implicit default.

Note that the mock is typically installed *after* `efa_test_resource_construct` and cleared in `TearDown`, so any wrapped call made while tearing the resources down (e.g. destroying the endpoint's `self_ah` via `ibv_destroy_ah`) is also subject to strict checking — expect those calls too.
