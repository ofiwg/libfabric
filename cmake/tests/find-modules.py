#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
"""Exercise find-module cache transitions and SDK selection with isolated metadata."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


cmake, generator, source, binary = sys.argv[1:]
source = Path(source)
work = Path(binary) / "find-modules-test"
shutil.rmtree(work, ignore_errors=True)
packages = (
    ("hwloc", "hwloc.h", "hwloc"),
    ("libibverbs", "infiniband/verbs.h", "ibverbs"),
    ("librdmacm", "rdma/rdma_cma.h", "rdmacm"),
    ("libefa", "infiniband/efadv.h", "efa"),
    ("uuid", "uuid/uuid.h", "uuid"),
    ("lttng-ust", "lttng/tracepoint.h", "lttng-ust"),
    ("liburing", "liburing.h", "uring"),
    ("json-c", "json-c/json.h", "json-c"),
    ("gdrcopy", "gdrapi.h", "gdrapi"),
    ("libze_loader", "level_zero/ze_api.h", "ze_loader"),
    ("hsa-runtime64", "hsa/hsa_ext_amd.h", "hsa-runtime64"),
)
for name in ("pkg-config-sdk", "override SDK"):
    root = work / name
    (root / "lib/pkgconfig").mkdir(parents=True)
    for pc, header, library in packages:
        path = root / "include" / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        (root / "lib" / f"lib{library}.a").touch()
        (root / "lib/pkgconfig" / f"{pc}.pc").write_text(
            f"Name: {pc}\nDescription: find-module fixture\nVersion: 2.5\n"
            f'Libs: -L"{root}/lib" -l{library}\n'
            f'Cflags: -I"{root}/include"\n'
        )
    for header in ("cuda.h", "cuda_runtime.h", "nvml.h"):
        (root / "include" / header).touch()

environment = dict(os.environ)
environment["PKG_CONFIG_PATH"] = ""
environment["PKG_CONFIG_LIBDIR"] = str(work / "pkg-config-sdk/lib/pkgconfig")
# Reuse each build directory deliberately: stale pkg-config cache entries must
# not stand in for imported targets in a subsequent configure run.
cases = (
    ("cached", "pkg-config-sdk", False, "2.5;EXACT", False),
    ("cached", "pkg-config-sdk", True, "", False),
    ("cached", "pkg-config-sdk", False, "2...<3", False),
    ("cached", "pkg-config-sdk", False, "999;EXACT", True),
    ("override", "override SDK", False, "", False),
    ("override", "override SDK", False, "2.5;EXACT", True),
)
for build, root, disabled, version, missing in cases:
    command = [
        cmake, "-S", str(source / "cmake/tests/find-modules"),
        "-B", str(work / build), "-G", generator,
        f"-DLIBFABRIC_SOURCE={source}", f"-DSDK_ROOT={work / root}",
        f"-DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig={'ON' if disabled else 'OFF'}",
        "-DCMAKE_DISABLE_FIND_PACKAGE_json-c=ON",
        "-DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=OFF",
        f"-DVERSION_ARGS={version}", f"-DEXPECT_MISSING={'ON' if missing else 'OFF'}",
    ]
    print(f"Checking {build}: pkg-config disabled={disabled}, version={version!r}",
          flush=True)
    subprocess.run(command, env=environment, check=True)
