# Static analysis with CMake

The `clang-tidy` and `iwyu` presets enable analysis alongside compilation.
Both use Ninja, select Clang for C, and inherit the debug build's provider
discovery. Install Ninja, Clang, and the selected analyzer first; normal
builds do not require either analyzer. Use tools and SDK headers compatible
with the compiler and target being analyzed.

```sh
cmake --preset clang-tidy
cmake --build --preset clang-tidy --parallel 4

cmake --preset iwyu
cmake --build --preset iwyu --parallel 4
```

These presets use CMake's standard `CMAKE_C_CLANG_TIDY` and
`CMAKE_C_INCLUDE_WHAT_YOU_USE` variables, and their `CXX` equivalents.
They apply to the enabled core, providers, utilities, and fabtests. Select
providers and accelerators with the usual `LIBFABRIC_*` options. Explicitly
enable providers when a missing SDK should fail configuration. Presets write
`compile_commands.json` for tools that consume a compilation database.

The same hooks work with Unix Makefiles. Visual Studio and Xcode generators
do not run these CMake analysis hooks. For a custom build, set the standard
variables directly; command arguments form a semicolon-separated list:

```sh
cmake -S . -B build/analysis -G Ninja -DCMAKE_C_COMPILER=clang \
  "-DCMAKE_C_CLANG_TIDY=clang-tidy;--config-file=$PWD/.clang-tidy" \
  '-DCMAKE_C_INCLUDE_WHAT_YOU_USE=include-what-you-use;-Xiwyu;--keep=*/config.h'
cmake --build build/analysis --parallel 4
```

An absolute executable path can select a versioned tool installation. Set
the corresponding `CMAKE_CXX_*` variables when analyzing C++ sources too.
The presets do not modify a parent project's settings. A parent may use the
same standard variables before `add_subdirectory(libfabric)`.

The `.clang-tidy` policy enables the Clang static analyzer and selected
bug-prone-operation checks. It excludes blanket suggestions to replace C
library functions with optional Annex K APIs such as `memcpy_s`. Findings
are advisory, and no automatic fixes are applied. The preset explicitly
passes the configuration file so generated sources in an external build
directory use the same policy. Override `CMAKE_C_CLANG_TIDY` to select other
checks or a different configuration.

IWYU preserves `config.h` because configuration macros and include order
can affect declarations even when no direct symbol use is apparent.
Suggestions still require review: platform wrappers, public umbrella
headers, SDK mappings, and conditional code can invalidate a proposed
removal. In particular, replacing `ofi_osd.h` with a platform's `osd.h`
would bypass the project's portability layer. Successful compilation does
not validate every IWYU recommendation or prove that analysis had no errors.

Analysis runs when objects compile. To rerun it after changing the tool's
configuration, use `cmake --build --preset clang-tidy --clean-first`, or the
corresponding IWYU command. CMake does not track analyzer configuration files
as compilation dependencies. Keep separate build directories for ordinary
and analysis builds.

## Initial experiment

The experiment used Clang/clang-tidy 23.1.1 and IWYU 0.26 (Clang 22.1.8) on
Linux x86-64. It selected all 18 providers and CUDA, ROCr, Level Zero, Neuron,
SynapseAI, and GDRCopy, using the SDKs recorded in the
[CMake validation snapshot](cmake.md#validation-snapshots). Accelerator
libraries were loaded at runtime. This exercised enabled source paths;
hardware transfers and other platform configurations were not tested.
Both analyzer builds and all eight CMake integration tests completed.
The curated clang-tidy policy reported 638 warnings at 424 distinct source
sites, including repeated header diagnostics and compiler warnings. IWYU
suggested additions or removals in 707 source/header files. These are
triage counts, not counts of confirmed defects.

Two clang-tidy findings were independently reproduced with AddressSanitizer
using the actual source functions and small harnesses. The reproducers passed
after applying the corresponding one-line fixes to temporary source copies:

| Location | Trigger and result |
| --- | --- |
| `prov/util/src/util_profile.c`, `ofi_prof_add_event` | Growing the callback array allocates `eventlist_size * sizeof(void *)`, although each `struct util_pcb` contains two pointers. Registering a callback in the upper half of the grown array produces a heap-buffer-overflow. |
| `prov/lnx/src/lnx_init.c`, `lnx_setup_core_fab` | A failed `fi_fabric` call for the second core fabric reaches `free(cf)`, where `cf` points inside `lf_core_fabrics`. AddressSanitizer reports an invalid free 16 bytes into the allocation on this platform. |

These source defects remain separate from the analysis integration. Another
diagnostic identifies an uninitialized `ze_ret` in `ze_hmem_copy` if a
previously initialized Level Zero command queue exists but command-list
allocation fails; that path still needs a runtime reproducer. Intrusive
lists and container recovery also produce diagnostics that require manual
inspection.

IWYU identified missing direct dependencies, such as `<strings.h>` for
`strcasecmp` in `src/perf.c` and `<unistd.h>` for `close` in
`src/hmem_neuron.c`. It also proposed platform-specific includes that should
not be applied mechanically. No source include lists were rewritten.

References: [CMake clang-tidy integration](https://cmake.org/cmake/help/latest/prop_tgt/LANG_CLANG_TIDY.html),
[CMake IWYU integration](https://cmake.org/cmake/help/latest/prop_tgt/LANG_INCLUDE_WHAT_YOU_USE.html),
and [IWYU documentation](https://github.com/include-what-you-use/include-what-you-use/blob/master/README.md).
