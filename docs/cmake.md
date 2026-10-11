# Building with CMake

CMake 3.21 or newer and a C11 compiler are required. A Git checkout can be
built directly; `autogen.sh`, Autoconf, Automake, and Libtool are not used.
Ninja and Makefile generators are supported. For example:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build
```

The default builds a shared library, utilities, and available providers.
Use `-DBUILD_SHARED_LIBS=OFF` for a static library. Both configurations export
one CMake target, `Libfabric::fabric`, and install `libfabric.pc`. The library
version and SONAME follow the existing release and ABI version definitions.

The `debug`, `release`, `static`, `minimal`, `asan`, and `ubsan` presets provide
starting configurations. For example, `cmake --preset minimal` selects TCP
and sockets. Presets put their output under `build/<preset>`.

With a multi-configuration generator, choose the configuration when building
and testing, for example `cmake --build build --config Debug` and
`ctest --test-dir build -C Debug --output-on-failure`.

## Provider selection

`LIBFABRIC_PROVIDER_<NAME>` accepts three values:

| Value | Behavior |
| --- | --- |
| `AUTO` | Build if the platform and dependencies allow it. |
| `ON` | Require the provider; fail configuration if it cannot be built. |
| `OFF` | Omit the provider and its dependency checks. |

Names are uppercase, for example `LIBFABRIC_PROVIDER_VERBS=ON` or
`LIBFABRIC_PROVIDER_PSM3=OFF`. The configuration output lists built-in and
loadable providers. Dependencies can be found through `CMAKE_PREFIX_PATH`,
pkg-config, or the corresponding `<Package>_ROOT` variable.
Providers default to `AUTO`, except UCX, which defaults to `OFF` to match
Autotools' opt-in behavior.

`LIBFABRIC_TCP_DLOPEN=ON` builds TCP as a separate provider plugin. The same
switch is available for other transport providers. Plugins require a shared
libfabric build. In the build tree, point `FI_PROVIDER_PATH` at
`build/libfabric/<configuration>`, for example `build/libfabric/Release`.
Installed plugins go in `<libdir>/libfabric`. If an installation is moved,
set `FI_PROVIDER_PATH` to that directory.

Hooks use `LIBFABRIC_HOOK_<NAME>` to select built-in support and
`LIBFABRIC_<NAME>_DLOPEN` for plugins, for example
`LIBFABRIC_TRACE_DLOPEN=ON`. The perf hook must remain built in: its current
implementation references private core state unavailable to plugins.
Autotools accepts `--enable-perf=dl` but the resulting plugin fails to load
with an unresolved `perf_domain` symbol. CMake rejects that configuration.

Libraries installed outside the system loader's search path may require
`CMAKE_INSTALL_RPATH` or `CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON`. These are
standard CMake settings; dependency search paths are not forced into the
installed runtime search path.

## Optional features

GPU backends are opt-in: `LIBFABRIC_ENABLE_CUDA`, `LIBFABRIC_ENABLE_ZE`,
`LIBFABRIC_ENABLE_ROCR`, `LIBFABRIC_ENABLE_NEURON`, and
`LIBFABRIC_ENABLE_SYNAPSEAI`. An enabled backend requires its development
files. CUDA, Level Zero, and ROCr default to runtime loading; their
`LIBFABRIC_<BACKEND>_DLOPEN` options select direct linking instead when OFF.

A CUDA build using runtime loading requires headers, including NVML, but no
CUDA compiler or installed driver. Use `CUDAToolkit_ROOT` as a CMake or
environment variable for a combined SDK (the CMake variable takes priority),
or set `LIBFABRIC_CUDA_INCLUDE_DIR`, `LIBFABRIC_CUDART_INCLUDE_DIR`, and
`LIBFABRIC_NVML_INCLUDE_DIR` for split header packages. Direct linking uses
CMake's `FindCUDAToolkit` and requires the CUDA driver, runtime, and NVML
libraries.

Level Zero, ROCr, and GDRCopy also need only headers in runtime-loading mode.
Their find modules expose a `Headers` component; default discovery requires
the runtime library for direct linking and installed static consumers.

PSM2's optional MQ fast paths default to OFF. PSM3 defaults to both transport
HALs, RC, and RDMA reads when its dependencies are available; UDP is opt-in.
Its CUDA and Level Zero backends require runtime loading, matching Autotools.
`LIBFABRIC_SHM_DSA`, `LIBFABRIC_PSM3_RV`, `LIBFABRIC_OPX_TARGET_CLONES`, and
`LIBFABRIC_OPX_HFISVC` accept `AUTO`, `ON`, or `OFF`. An explicit `ON` requires
the feature's dependencies. `LIBFABRIC_OPX_TRACER` accepts comma-separated
categories such as `TX,RX` or `all,^RELI`. Alternate kernel or SDK headers can
be supplied through `DSA_ROOT`, `RV_ROOT`, `OPXS_ROOT`, and `OPXHfiSvc_ROOT`.
OPX with CUDA or ROCr requires both `sdma_req_meminfo` in `hfi1_user.h` and
the TID-update-v3 API: `hfi1_tid_info_v3` in `hfi1_ioctl.h` and
`HFI1_IOCTL_TID_UPDATE_V3` in `rdma_user_ioctl.h`. Set `OPXS_ROOT` to the
matching kernel header installation. OPX uses the core ROCr backend and does
not require HIP.

Memory monitors, XPMEM, and io_uring are enabled when their dependencies are
available. They can be disabled with the corresponding `LIBFABRIC_*` cache
options. `LIBFABRIC_DEFAULT_MONITOR` selects `memhooks`, `uffd`, `kdreg2`, or
`disabled`; an unavailable selection is rejected.

`LIBFABRIC_BUILD_FABTESTS=ON` builds fabtests. `LIBFABRIC_BUILD_UTILITIES=OFF`
omits the command-line utilities. `CMAKE_INTERPROCEDURAL_OPTIMIZATION`,
`LIBFABRIC_ENABLE_ASAN`, `LIBFABRIC_ENABLE_TSAN`, `LIBFABRIC_ENABLE_UBSAN`,
and `LIBFABRIC_ENABLE_LSAN` control compiler instrumentation.
Configuration-specific IPO settings such as
`CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE` are also supported. The public
API translation units are excluded from IPO because their symbol-version
assembly must stay with the corresponding function definitions.

`LIBFABRIC_ENABLE_COVERAGE=ON` enables gcov-compatible instrumentation with
optimization disabled and debug information enabled. Required runtime link
flags are propagated to installed CMake and pkg-config consumers.
`LIBFABRIC_THREAD_SAFETY_ANALYSIS=ON` enables Clang's lock annotations and
thread-safety warnings when supported; other compilers issue a warning.
`LIBFABRIC_PICKY=ON` includes Autotools' developer warning set.

`LIBFABRIC_ENABLE_ATOMICS=OFF` selects mutex emulation on POSIX platforms.
Providers that require native atomics become unavailable; explicitly
requesting one fails configuration. With atomics enabled, discovery checks
both 32-bit and 64-bit operations and links libatomic when needed. To disable
epoll or kqueue, use the standard `CMAKE_DISABLE_FIND_PACKAGE_Epoll=ON` or
`CMAKE_DISABLE_FIND_PACKAGE_Kqueue=ON` setting.

Fabtests includes the component diagnostics, Level Zero DMA-BUF executables,
and LPP/verbs/EFA suites when their prerequisites are available. Provider
suites can be disabled independently with `LIBFABRIC_FABTESTS_<PROVIDER>=OFF`.
LPP's ROCr regression helper additionally requires HIP headers; a split SDK
can set `LIBFABRIC_FABTESTS_HIP_INCLUDE_DIR`. The installed Python runner needs
its normal Python dependencies and the standard `bin`/`share` layout.

Direct provider builds, embedded mode, bundled PSM2, external PSM3, the EFA
provider's unit-test suite, CXI developer tests, and the EFA GDA test are not implemented
by this CMake build. The existing
Autotools and native Windows build files remain available.

## Using libfabric from CMake

After installation:

```cmake
find_package(Libfabric CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE Libfabric::fabric)
```

Set `CMAKE_PREFIX_PATH` when installing outside a standard prefix. A static
installation rediscovers its private dependencies on the consuming machine.
Private compiler definitions and build-tree paths are not exported.
Dependencies retain stable target names whether found through pkg-config,
a vendor config package, or a fallback find module. Curl uses CMake's
`FindCURL`; json-c and ROCr prefer their upstream config packages.

For pkg-config consumers, `libfabric.pc` records dependency-owned `.pc`
packages in `Requires.private` so pkg-config resolves their transitive link
flags. Set `PKG_CONFIG_PATH` for SDKs installed outside standard prefixes.
SDK runtimes without `.pc` metadata are listed explicitly in `Libs.private`.
Using `pkg-config --static` also requires the development libraries named
by those packages' static metadata, such as libudev for hwloc on Ubuntu.

A parent project can also use `add_subdirectory(path/to/libfabric)` and link
the same target. Installation, utilities, and CMake integration tests default
to OFF when libfabric is a subproject. Explicit cache options override those
defaults.

## Validation and maintenance

CTest covers provider discovery, TCP payload verification, relocated C and
C++ consumers, and provider reconfiguration and error handling. The TCP test
uses Python 3 when available. Install tests require a C++ compiler in addition
to the library's C compiler. Disable the integration tests with
`LIBFABRIC_BUILD_TESTING=OFF` when those tools are unavailable.
The integration suite runs only for native builds; cross builds do not
register tests that execute target binaries or configure host consumers.
Relocation tests use a prefix containing spaces. Static consumers also test
recovery from a failed dependency lookup and discovery without pkg-config
or vendor config packages.

On Unix hosts with Python and pkg-config, isolated SDK fixtures also check
dependency version constraints, explicit SDK roots, and reconfiguration
after disabling pkg-config. These tests cover discovery and generation;
the installed-consumer tests exercise real dependency linking. On Linux,
synthetic EFA SDKs also exercise completion-action feature detection with
each prerequisite absent and with the complete API.
On supported OPX architectures, synthetic kernel SDKs check each missing
HMEM interface and complete support without a HIP dependency. CUDA discovery
fixtures exercise the environment root with headers alone. The parent-project
test supplies unrelated compiler-probe settings to ensure libfabric isolates
its feature checks and preserves the caller's state.
CMake 3.21's upstream `FindPkgConfig` mishandles include paths containing
spaces; keep pkg-config SDK prefixes free of spaces with that release.

Source lists are explicit. When adding or removing a source, update both its
`Makefile.include` and `CMakeLists.txt`. Package discovery belongs in
`cmake/Find*.cmake`; library feature probes belong in the build configuration,
so installed consumers do not repeat libfabric's compiler checks. Each
provider owns its include paths, feature macros, and architecture flags.
Use CMake's standard checks directly for one-use feature probes. Functions
with keyword arguments use `cmake_parse_arguments(PARSE_ARGV ...)` to preserve
quoted lists. Find modules use pkg-config as a hint for CMake's native search;
its version and transitive requirements apply only to the selected library.
Every discovery path goes through `find_package_handle_standard_args` so
version constraints and optional lookups have consistent behavior.
When adding a link dependency, describe its CMake package and pkg-config
requirements in `cmake/LibfabricPackageDependencies.cmake`. Missing entries
are configuration errors. Only dependencies linked into the library are
exported; plugin dependencies remain private to the plugins.

The project formatting policy is in `.cmake-format.yaml`: 80 columns,
two-space indentation, preserved command spelling and argument order, and
explicit signatures for every project function and macro. Install the
formatter and run the same checks as CI:

```sh
python3 -m pip install -r cmake/requirements.txt
python3 cmake/format.py --in-place
python3 cmake/format.py --check
```

The formatter version is maintained in `cmake/requirements.txt`. Dependabot
checks it weekly alongside the repository's existing GitHub Actions updates.
Formatter updates run the CMake workflow, including formatting and custom
command validation, before review.

The checker parses all maintained, tracked `CMakeLists.txt`, `.cmake`, and
`.cmake.in` files, excluding vendored GoogleTest. It rejects unknown commands,
missing registrations, and signature changes that disagree with the actual
function declarations and `cmake_parse_arguments` definitions. Update the
registry when changing a custom command's arguments.

Module ownership and inclusion rules are documented in each module's `.rst`
header. The main `CMakeLists.txt` assembles targets; release/ABI parsing belongs
in `LibfabricVersion`, and source inventories belong in `LibfabricSources`.
`LibfabricOptions` declares user choices, while `LibfabricTargetSettings`
defines functions that apply private compile and instrumentation settings.
`LibfabricDependencies` dispatches searches, `LibfabricHmemDependencies`
handles accelerator SDK selection, and `Check<Package>` modules own SDK API
probes. `Find<Package>` modules remain usable by installed consumers without
running libfabric's provider checks.

Function-definition modules have global include guards. Build setup modules
use variable-scope guards because their results belong to the calling build
directory. Find modules and the installed package entry point deliberately
have no guard: subsequent calls may change versions, components, or dependency
hints. Check modules also remain unguarded to populate the current caller's
scope; individual compiler probes already use CMake's internal cache.
The SDK fixtures test same-configure version retries and Headers-to-Runtime
component transitions.

Low-level include/library paths and the test C++ compiler are advanced cache
entries. Provider and feature choices remain visible. SDK roots are public
discovery hints, and compiler-probe results remain internal cache entries.

Compilation does not establish hardware correctness. Provider execution and
GPU transfers still require their respective hardware test environments.

## Validation snapshots

### Refresh against main (2026-10-16)

All 18 providers were rebuilt together in shared, static, and plugin modes
against upstream `24fa1dcfaa5c`. Their CTest suites passed, including
installation, relocated consumers, and TCP payload checks. The shared build
also compiled fabtests. SDKs outside the system runtime search path used
`CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON`.

CUDA 12.8, ROCr 6.4.3, Level Zero 1.26.3, Neuron 2.34.10, SynapseAI, and
GDRCopy 2.5.1 built together with runtime loading and all 18 providers.
OPX used matching UAPI headers from Cornelis' `opa-distro-drivers`
`driver-ubuntu24.04-12_1` branch at
`15337de75d73aa6bbd0c9dede74801a20d6d8d64`. A separate OPX/ROCr build also
passed without HIP. The combined accelerator build and the OPX/ROCr build
passed their CTest suites; accelerator transfers were not exercised.
SynapseAI headers came from SynapseAI_Core revision
`d4d7f5c6f9f9cfc34d2a97b3773901d15443d4ee`.

The configuration regressions cover inherited parent-project probe state,
CUDA SDK roots supplied through the environment, native-test exclusion in
cross builds, and four complete/partial OPX kernel SDK configurations. The
parent-project test preserves atomics and symbol versioning while leaving
its caller's temporary check settings unchanged. OPX rejects missing SDMA,
TID structure, or ioctl interfaces during configuration. An unused HIP link
dependency has been removed from both provider and installed metadata.

EFA now includes all 70 C sources from its Autotools manifest and checks all
four prerequisites for completion actions. Five synthetic SDK configurations
verify the complete feature and each missing prerequisite independently.
These fixtures validate feature selection, not the completion-action data
path. EFA and verbs also built and passed install/consumer and TCP tests
with rdma-core 57.0 and 65.0, supplementing the all-provider build with 60.0.
None of these installed SDKs exposes the completion-action APIs.

### Broader option coverage (2026-10-11)

On Linux x86-64, all 18 transport providers were explicitly required and
built together as built-ins in shared and static libraries, and as loadable
plugins: TCP, UDP, sockets, RxM, RxD, mrail, lnx, sm2, shm, verbs, EFA, PSM2,
PSM3, OPX, CXI, UCX, usNIC, and LPP. The seven hooks were built in. This used
rdma-core 60.0, liburing 2.12, libpsm2 12.0.1, libcxi/Cassini headers 13.0.0,
and UCX 1.18.1. Installation, relocated C/C++ consumers, pkg-config consumers,
and TCP loopback checks passed for these configurations. EFA fabtests also
built with rdma-core 60.0.

The existing UCX source uses the four-argument `ucs_memtype_cache_update`
API. UCX 1.22 changed that signature; CMake detects the incompatibility at
configuration time. Supporting that API requires a separate provider source
change. PSM3's Level Zero backend also requires DRM development headers.

| HMEM interface | Compile/link coverage |
| --- | --- |
| System memory | TCP payload verification and consumer execution |
| CUDA | CUDA 12.8 headers with runtime loading; direct linking with NVIDIA SDK stub driver/NVML libraries |
| ROCr | ROCm 6.4.3, runtime loading and direct linking |
| Level Zero | Level Zero 1.26.3, runtime loading and direct linking; PSM3 with DRM headers |
| Neuron | Neuron runtime 2.34.10 headers, runtime loading |
| SynapseAI | Public SynapseAI headers, runtime loading |

GDRCopy 2.5.1 was also built with runtime loading and direct linking. CUDA,
Level Zero, ROCr, and GDRCopy also built together from a header-only SDK
prefix. PSM3 was checked with CUDA and Level Zero together. The all-provider
plugin build uses system memory. OPX with CUDA or ROCr requires vendor kernel
headers; the current check requires SDMA and TID-update-v3 interfaces. The
newer validation snapshot above covers those headers and OPX accelerator
compilation. Autotools currently checks only `sdma_req_meminfo`, which can
accept an incomplete header installation. Linux CI requires
17 providers (CXI needs a separate vendor SDK), and covers shared/static
libraries, plugins, fabtests, and Clang ASan/UBSan. Native Windows and macOS
jobs run the install, consumer, and loopback tests. CMake 3.21.4, LTO, Ninja Multi-Config,
and `add_subdirectory` were exercised locally. FreeBSD and hardware-specific
transfers were not tested. These results do not cover every combination of
provider, backend, SDK release, architecture, and build option.

## Comparison with Autotools

Each provider's `configure.m4` and `Makefile.include` was compared with its
CMake definition for source selection, private flags, dependencies, feature
probes, platform gates, defaults, and built-in/plugin behavior. The table
records the resulting checks and corrections; it does not assert hardware
execution coverage.

| Provider | Comparison result |
| --- | --- |
| TCP | All 17 C sources and common dependencies agree. |
| UDP | All 6 C sources agree; socket support follows the platform requirements. |
| sockets | All 23 C sources agree; common discovery supplies `shm_open` and any `librt` requirement. |
| RxM | All 14 C sources agree, including HMEM handling. |
| RxD | All 12 C sources and dependencies agree. |
| mrail | All 8 C sources and dependencies agree. |
| LNX | All 8 C sources agree; removed an unsupported Linux-only restriction. |
| SHM | All 15 C sources agree; restored atomic/CMA gates and header-only DSA discovery with an explicit feature switch. |
| SM2 | All 13 C sources agree; restored atomic/CMA gates. |
| LPP | All 26 C sources agree; restored C11 atomics requirement and effective thread-safety selection. |
| verbs | All 15 C sources agree; restored hwloc affinity, baseline SDK probing, and dependency ordering. |
| EFA | All 70 C sources agree; baseline/counter/completion-action probes, dependency ordering, and SynapseAI prerequisites match. |
| CXI | All 35 production C sources agree; developer tests remain outside this build. |
| usNIC | Restored private API macros, header-only verbs dependency, and matched libnl ABI discovery; removed an extra source absent from Autotools. |
| UCX | All 13 C sources agree; restored opt-in default and removed an unsupported Linux-only restriction. |
| PSM2 | All 18 external-library C sources agree; restored optional fast-path gates, baseline API check, dependency-based platform selection, and plugin filename. |
| PSM3 | Accounted for all 91 compiled C sources; corrected RC/RDMA-read/UDP controls, kernel DSA discovery, RV compatibility, and GPU source gates. |
| OPX | All 46 C sources agree; restored native-atomic, target-clone, DMA-BUF, HFISVC, GPU-header, and tracer controls. |
| Hooks | All seven hook source sets agree; six plugins build and load, while perf remains built in. |

CMake omits inactive PSM3 HAL/GPU translation units that Autotools compiles as
preprocessor stubs. It installs all shipped provider manuals. Standard CMake
RPATH settings replace Autotools' provider-specific prefix handling. GPU
backends are explicitly selected and default to runtime loading; Autotools
auto-discovers them and defaults to direct linking. These are deliberate
interface differences. PSM3 revision metadata uses `LIBFABRIC_BUILD_ID` and
`SOURCE_DATE_EPOCH` rather than Autotools' source checksum and build time.
The usNIC legacy fake driver remains excluded. CXI uses the normal dynamic
loader search instead of Autotools' explicit curl/json installation-prefix
overrides. Non-Linux PSM2/UCX and vendor OPX HFISVC/RV paths still need their
respective SDK and platform environments. OPX GPU compilation is covered by
the current validation snapshot; hardware execution remains untested.

The broader configure/build/install comparison also found gaps outside the
provider definitions:

| Autotools capability | CMake status |
| --- | --- |
| `--enable-coverage` | Supported by `LIBFABRIC_ENABLE_COVERAGE`, including consumer runtime linking. |
| `--enable-thread-safety-analysis` | Supported by `LIBFABRIC_THREAD_SAFETY_ANALYSIS`, including `OFI_THREAD_SAFETY_ANALYSIS`. |
| `--enable-picky` | Includes missing-prototype, strict-prototype, comment, and pedantic warnings. |
| `--disable-atomics`, libatomic fallback | Supports core lock emulation and library-backed 32/64-bit atomics. |
| `--disable-epoll`, `--disable-kqueue` | Use CMake's standard package-disable switches. |
| CUDA PCIe DMA-BUF flag in fabtests | Probed from SDK declarations and defined for fabtests. |
| Component, regression, LPP, and verbs fabtests | Source/target lists restored, including `fi_check_hmem`, five Level Zero diagnostics, and the `sighandler_test` name expected by pytest. |
| Installed fabtests runner and data | Installs runners in `bin`, pytest and exclusions in `share/fabtests`, and manuals in their correct sections. |
| Standalone fabtests against an installed libfabric | Not implemented; fabtests uses the library and backend configuration from the same CMake build. |
| Shared and static libraries from one configuration | Not implemented; use separate CMake build directories. |
| `--enable-only` | No single equivalent switch; select providers explicitly or start with the minimal preset. |
| `--without-dlopen` | No global disable switch; POSIX builds currently require dlopen. |
| `--enable-direct`, `--enable-embedded` | Not implemented. Subproject installation defaults do not reproduce embedded symbol-versioning behavior. |
| Bundled PSM2, external PSM3, EFA/CXI unit tests, EFA GDA | Not implemented. |
| Debug/compiler flag defaults | CMake configuration/toolchain defaults apply; Autotools' default warning and Debug stack-protector flags are not applied automatically. |
| Maintainer distribution/manual targets | No CMake equivalents of `distcheck` or `nroff`; Autotools packages the CMake sources, and CMake installs shipped manuals. |

These are capability gaps or explicit interface differences, rather than a
claim of complete Autotools parity. The literal configuration-macro comparison
also checked EFA and CXI names generated by CMake loops; those are present.

The gap fixes were checked with shared/static coverage builds, Clang thread
safety analysis and picky warnings, a TCP/sockets build with native atomics
and epoll disabled, and a static libatomic-backed build. On GCC 16, the latter
used `-fno-inline-atomics -fno-link-libatomic` so the compiler could not hide
missing explicit link requirements. Both native probes failed, library-backed
probes passed, and installed metadata included `-latomic`. Consumer and TCP
payload tests passed. Coverage execution produced `.gcda` files. The installed
Python fabtests runner also passed `--help` after relocating its prefix.
GCC's implicit libatomic dependency is exported too: a Clang consumer linked
and ran against the GCC-produced static archive through pkg-config. All five
Level Zero component diagnostics compiled with rdma-core 60.0; LPP regression
helpers compiled with CUDA and ROCr enabled using HIP 6.4.3 headers.

## Build measurements

The repeatable comparison is:

```sh
python3 cmake/benchmarks/build.py --jobs 8 --runs 3 \
  --output /path/to/empty/results
```

Run it from a committed Git checkout with CMake, Ninja, Autoconf, Automake,
Libtool, Make, GCC, binutils, and Python installed. It snapshots the source
separately for each build, alternates build order, and writes command logs,
symbol lists, and JSON measurements. Source copying and binary inspection
are outside the timed intervals. Filesystem caches are not flushed.

The following medians were measured on 2026-10-16 from source commit
`c3eca6912825`, before the later probe-isolation, SDK-discovery, and OPX
prerequisite corrections. The host was an AMD Ryzen 7 9800X3D under WSL2,
using GCC 16.2.1,
CMake 4.4.4, Ninja 1.13.2, GNU Make 4.4.1, and Autoconf 2.73. Both builds
use `-O2 -DNDEBUG`, eight jobs, a shared library, TCP and sockets, and three
utilities. Hooks, accelerator backends, io_uring, and XPMEM are disabled.
Autotools is bootstrapped with `autoreconf -fi`; its build disables the static
library. The current Autotools 128-bit atomic probe fails because it uses
`NULL` without including `<stddef.h>`. The CMake probe result is explicitly
disabled for this comparison to match the generated feature set.

| Measurement | CMake + Ninja | Autotools + Make |
| --- | ---: | ---: |
| Bootstrap | 0.00 s | 6.25 s |
| Configure | 4.48 s | 9.28 s |
| Compile/link | 2.85 s | 3.13 s |
| Complete clean build | 7.27 s | 18.65 s |
| No-op rebuild | 0.0042 s | 0.1514 s |
| Rebuild after touching `xnet_ep.c` | 0.362 s | 0.651 s |
| Stripped shared library | 761,608 bytes | 732,928 bytes |

Each clean-build value is the median of three runs; no-op and incremental
values use five and three repetitions per build respectively. Independent
phase medians need not sum to the total median. Clean-build ranges were
7.00–7.33 s and 18.55–19.63 s. Raw results are in
[results.json](../cmake/benchmarks/results.json).

This configuration built about 2.6 times faster overall. The CMake binary
was 3.9% larger, and all six builds had the same 54 defined dynamic-symbol
entries, including version nodes. There is no demonstrated binary-size or
runtime-throughput improvement. These timings describe this provider set
and machine, not every deployment.
